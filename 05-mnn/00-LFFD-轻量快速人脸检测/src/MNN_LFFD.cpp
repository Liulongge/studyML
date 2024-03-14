
#include "MNN_LFFD.h"

const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
const float norm_vals[3] = {0.0078431373, 0.0078431373, 0.0078431373};

LFFD::LFFD(const std::string &model_path, int scale_num, int num_thread_)
{
	num_output_scales_ = scale_num;
	num_thread_ = num_thread_;
	outputTensors_.resize(scale_num * 2);
	if (num_output_scales_ == 5)
	{

		mnn_model_file_ = model_path;
		receptive_field_list_ = {20, 40, 80, 160, 320};
		receptive_field_stride_ = {4, 8, 16, 32, 64};
		// bbox_small_list_ = {10, 20, 40, 80, 160};
		// bbox_large_list_ = {20, 40, 80, 160, 320};
		receptive_field_center_start_ = {3, 7, 15, 31, 63};

		for (int i = 0; i < receptive_field_list_.size(); i++)
		{
			constant_.push_back(receptive_field_list_[i] / 2);
		}

		output_blob_names_ = {"softmax0", "conv8_3_bbox",
							  "softmax1", "conv11_3_bbox",
							  "softmax2", "conv14_3_bbox",
							  "softmax3", "conv17_3_bbox",
							  "softmax4", "conv20_3_bbox"};
	}
	else if (num_output_scales_ == 8)
	{
		mnn_model_file_ = model_path + "/symbol_10_560_25L_8scales_v1_deploy.mnn";
		receptive_field_list_ = {15, 20, 40, 70, 110, 250, 400, 560};
		receptive_field_stride_ = {4, 4, 8, 8, 16, 32, 32, 32};
		// bbox_small_list_ = {10, 15, 20, 40, 70, 110, 250, 400};
		// bbox_large_list_ = {15, 20, 40, 70, 110, 250, 400, 560};
		receptive_field_center_start_ = {3, 3, 7, 7, 15, 31, 31, 31};

		for (int i = 0; i < receptive_field_list_.size(); i++)
		{
			constant_.push_back(receptive_field_list_[i] / 2);
		}

		output_blob_names_ = {"softmax0", "conv8_3_bbox",
							  "softmax1", "conv10_3_bbox",
							  "softmax2", "conv13_3_bbox",
							  "softmax3", "conv15_3_bbox",
							  "softmax4", "conv18_3_bbox",
							  "softmax5", "conv21_3_bbox",
							  "softmax6", "conv23_3_bbox",
							  "softmax7", "conv25_3_bbox"};
	}

	lffd_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_model_file_.c_str()));
	MNN::ScheduleConfig config;
	config.type = MNN_FORWARD_CPU;
	config.numThread = num_thread_;

	MNN::BackendConfig backendConfig;
	backendConfig.precision = MNN::BackendConfig::Precision_High;
	backendConfig.power = MNN::BackendConfig::Power_High;
	config.backendConfig = &backendConfig;

	sess_lffd_ = lffd_->createSession(config);
	input_tensor_ = lffd_->getSessionInput(sess_lffd_, NULL);
	for (int i = 0; i < output_blob_names_.size(); i++)
	{
		outputTensors_[i] = lffd_->getSessionOutput(sess_lffd_, output_blob_names_[i].c_str());
	}

	::memcpy(img_config_.mean, mean_vals, sizeof(mean_vals));
	::memcpy(img_config_.normal, norm_vals, sizeof(norm_vals));

	img_config_.sourceFormat = (MNN::CV::ImageFormat)2;
	img_config_.destFormat = (MNN::CV::ImageFormat)2;

	img_config_.filterType = (MNN::CV::Filter)(2);
	img_config_.wrap = (MNN::CV::Wrap)(2);
}

LFFD::~LFFD()
{
	lffd_->releaseModel();
	lffd_->releaseSession(sess_lffd_);
}

int LFFD::detect(cv::Mat &img, std::vector<FaceInfo> &face_list, int resize_h, int resize_w,
				 float score_threshold, float nms_threshold, int top_k, std::vector<int> skip_scale_branch_list)
{

	if (img.empty())
	{
		std::cout << "image is empty ,please check!" << std::endl;
		return -1;
	}

	image_h_ = img.rows;
	image_w_ = img.cols;

	cv::Mat in;
	cv::resize(img, in, cv::Size(resize_w, resize_h));
	float ratio_w = (float)image_w_ / resize_w;
	float ratio_h = (float)image_h_ / resize_h;

	// resize session and input tensor
	std::vector<int> inputDims = {1, 3, resize_h, resize_w};
	std::vector<int> shape = input_tensor_->shape();
	shape[0] = 1;
	shape[2] = resize_h;
	shape[3] = resize_w;
	lffd_->resizeTensor(input_tensor_, shape);
	lffd_->resizeSession(sess_lffd_);

	// prepare data
	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(img_config_));
	pretreat->convert(in.data, resize_w, resize_h, in.step[0], input_tensor_);

	// forward
	lffd_->runSession(sess_lffd_);

	std::vector<FaceInfo> bbox_collection;
	for (int i = 0; i < num_output_scales_; i++)
	{
		MNN::Tensor *tensor_score = new MNN::Tensor(outputTensors_[2 * i], MNN::Tensor::CAFFE);
		outputTensors_[2 * i]->copyToHostTensor(tensor_score);

		MNN::Tensor *tensor_location = new MNN::Tensor(outputTensors_[2 * i + 1], MNN::Tensor::CAFFE);
		outputTensors_[2 * i + 1]->copyToHostTensor(tensor_location);

		generateBBox(bbox_collection, tensor_score, tensor_location, score_threshold,
					 tensor_location->width(), tensor_location->height(), img.cols, img.rows, i);

		delete tensor_score;
		delete tensor_location;
	}
	std::vector<FaceInfo> valid_input;
	get_topk_bbox(bbox_collection, valid_input, top_k);
	nms(valid_input, face_list, nms_threshold);

	for (int i = 0; i < face_list.size(); i++)
	{
		/* 根据图像预处理时resize比率, 将其还原 */
		face_list[i].x0 *= ratio_w;
		face_list[i].y0 *= ratio_h;
		face_list[i].x1 *= ratio_w;
		face_list[i].y1 *= ratio_h;

		/* 越界保护 */
		float w, h, maxSize;
		float cx, cy;
		w = face_list[i].x1 - face_list[i].x0;
		h = face_list[i].y1 - face_list[i].y0;

		maxSize = w > h ? w : h;
		cx = face_list[i].x0 + w / 2;
		cy = face_list[i].y0 + h / 2;
		face_list[i].x0 = cx - maxSize / 2 > 0 ? cx - maxSize / 2 : 0;
		face_list[i].y0 = cy - maxSize / 2 > 0 ? cy - maxSize / 2 : 0;
		face_list[i].x1 = cx + maxSize / 2 > image_w_ ? image_w_ - 1 : cx + maxSize / 2;
		face_list[i].y1 = cy + maxSize / 2 > image_h_ ? image_h_ - 1 : cy + maxSize / 2;
	}
	return 0;
}

void LFFD::generateBBox(std::vector<FaceInfo> &bbox_collection, MNN::Tensor *tensor_score, MNN::Tensor *tensor_location, float score_threshold,
						int out_tensor_w, int out_tensor_h, int img_w, int img_h, int scale_id)
{
	/* 根据网路预先定义的信息, 计算各个anchor box的中心点坐标
	 * 网络输入size确定后, 该信息已确定 */
	float RF_cx[out_tensor_w];
	float RF_cx_mat[out_tensor_w * out_tensor_h];
	float RF_cy[out_tensor_h];
	float RF_cy_mat[out_tensor_h * out_tensor_w];

	for (int x = 0; x < out_tensor_w; x++)
	{
		RF_cx[x] = receptive_field_center_start_[scale_id] + receptive_field_stride_[scale_id] * x;
	}
	for (int x = 0; x < out_tensor_h; x++)
	{
		for (int y = 0; y < out_tensor_w; y++)
		{
			RF_cx_mat[x * out_tensor_w + y] = RF_cx[y];
		}
	}

	for (int x = 0; x < out_tensor_h; x++)
	{
		RF_cy[x] = receptive_field_center_start_[scale_id] + receptive_field_stride_[scale_id] * x;
		for (int y = 0; y < out_tensor_w; y++)
		{
			RF_cy_mat[x * out_tensor_w + y] = RF_cy[x];
		}
	}


	/* 结合anchor box信息, 根据网络回归分支预测的偏移量, 计算每个预测框的位置 */
	float x1_mat[out_tensor_h * out_tensor_w];
	float y1_mat[out_tensor_h * out_tensor_w];
	float x2_mat[out_tensor_h * out_tensor_w];
	float y2_mat[out_tensor_h * out_tensor_w];

	float *box_map_ptr = tensor_location->host<float>();
	// x-left-top
	int fea_spacial_size = out_tensor_h * out_tensor_w;
	for (int j = 0; j < fea_spacial_size; j++)
	{
		float x0 = RF_cx_mat[j] - box_map_ptr[0 * fea_spacial_size + j] * constant_[scale_id];
		x1_mat[j] = x0 < 0 ? 0 : x0;
	}
	// y-left-top
	for (int j = 0; j < fea_spacial_size; j++)
	{
		float y0 = RF_cy_mat[j] - box_map_ptr[1 * fea_spacial_size + j] * constant_[scale_id];
		y1_mat[j] = y0 < 0 ? 0 : y0;
	}
	// x-right-bottom
	for (int j = 0; j < fea_spacial_size; j++)
	{
		float x1 = RF_cx_mat[j] - box_map_ptr[2 * fea_spacial_size + j] * constant_[scale_id];
		x2_mat[j] = x1 > img_w - 1 ? img_w - 1 : x1;
	}
	// y-right-bottom
	for (int j = 0; j < fea_spacial_size; j++)
	{
		float y1 = RF_cy_mat[j] - box_map_ptr[3 * fea_spacial_size + j] * constant_[scale_id];
		y2_mat[j] = y1 > img_h - 1 ? img_h - 1 : y1;
	}


	/* 根据预测分支的得分, 过滤得分过低的box */
	float *score_map_ptr = tensor_score->host<float>();
	for (int k = 0; k < fea_spacial_size; k++)
	{
		if (score_map_ptr[k] > score_threshold)
		{
			FaceInfo face_info;
			face_info.x0 = x1_mat[k];
			face_info.y0 = y1_mat[k];
			face_info.x1 = x2_mat[k];
			face_info.y1 = y2_mat[k];
			face_info.score = score_map_ptr[k];
			face_info.area = (face_info.x1 - face_info.x0) * (face_info.y1 - face_info.y0);
			bbox_collection.push_back(face_info);
		}
	}
}

void LFFD::get_topk_bbox(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int top_k)
{
	std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b)
			  { return a.score > b.score; });

	if (input.size() > top_k)
	{
		for (int k = 0; k < top_k; k++)
		{
			output.push_back(input[k]);
		}
	}
	else
	{
		output = input;
	}
}

void LFFD::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float threshold, int type)
{
	if (input.empty())
	{
		return;
	}
	/* 根据得分, 由高到低排序 */
	std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b)
			  { return a.score > b.score; });
	int box_num = input.size();
	std::vector<int> merged(box_num, 0);
	/* box两两进行比较, 面积重合度高的仅保留得分高的box(排在前面的)
	 * 从得分高的box入手, 抑制后续box */
	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			/* 求两个box的交集面积 */
			float inner_x0 = input[i].x0 > input[j].x0 ? input[i].x0 : input[j].x0; // std::max(input[i].x0, input[j].x0);
			float inner_y0 = input[i].y0 > input[j].y0 ? input[i].y0 : input[j].y0;
			float inner_x1 = input[i].x1 < input[j].x1 ? input[i].x1 : input[j].x1; // bug fixed ,sorry
			float inner_y1 = input[i].y1 < input[j].y1 ? input[i].y1 : input[j].y1;
			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;
			if (inner_h <= 0 || inner_w <= 0)
				continue;
			float inner_area = inner_h * inner_w;

			/* 求其他box原始面积 */
			float h1 = input[j].y1 - input[j].y0 + 1;
			float w1 = input[j].x1 - input[j].x0 + 1;
			float area1 = h1 * w1;

			/* 交集与原始面积比值, 若比值大于阈值, 则证明box[j]举例box[i]过近, 丢弃box[j] */
			float score = inner_area / area1;
			if (score > threshold)
				merged[j] = 1;
		}
	}
}
