#include <iostream>
#include "MNN_LFFD.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <chrono>
#include "3dparty/cmdline.h"
using namespace cv;

int main(int argc, char **argv)
{
	/* 参数解析 */
	cmdline::parser cmd;
	cmd.add<std::string>("model_name", 'm', "model name", false, "../models/symbol_10_320_20L_5scales_v2_deploy.mnn");
	cmd.add<std::string>("data_name", 'd', "data name", false, "../data/640.jpeg");
	cmd.add<std::string>("run_mode", 'r', "run mode[offline/online]", true, "offline", cmdline::oneof<std::string>("offline", "online"));
	cmd.add<int>("scale_num", 's', "scale num[5/8]", false, 5, cmdline::oneof<int>(5, 8));
	cmd.add("help", 'h', "help info");
	bool ok = cmd.parse(argc, argv);
	if (!ok)
	{
		std::cout << cmd.error() << std::endl
				  << cmd.usage();
		return 0;
	}

	std::string model_name = cmd.get<std::string>("model_name");
	std::string data_name = cmd.get<std::string>("data_name");
	std::string run_mode = cmd.get<std::string>("run_mode");
	int scale_num = cmd.get<int>("scale_num");

	/* 实例化 */
	LFFD *face_detector = new LFFD(model_name, scale_num, 8);
	if (run_mode == "offline")
	{
		cv::Mat image = cv::imread(data_name);
		std::vector<FaceInfo> finalBox;

		std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
		face_detector->detect(image, finalBox, image.rows, image.cols);
		std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
		std::cout << "mtcnn time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000 << "ms" << std::endl;

		for (int i = 0; i < finalBox.size(); i++)
		{
			FaceInfo facebox = finalBox[i];
			cv::Rect box = cv::Rect(facebox.x0, facebox.y0, facebox.x1 - facebox.x0, facebox.y1 - facebox.y0);
			cv::rectangle(image, box, cv::Scalar(255, 0, 21), 2);
		}
		std::cout << "box num: " << finalBox.size() << std::endl;
		cv::imwrite("res.jpg", image);
		cv::namedWindow("MNN", WINDOW_NORMAL);
		cv::imshow("MNN", image);
		cv::waitKey();
	}
	else
	{
		cv::VideoCapture cap(0);
		cv::Mat image;
		while (1)
		{
			cap >> image;
			std::vector<FaceInfo> finalBox;
			std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
			face_detector->detect(image, finalBox);
			std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
			float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
			std::cout << "lffd time:" << dur << "ms" << std::endl;

			for (int i = 0; i < finalBox.size(); i++)
			{
				FaceInfo facebox = finalBox[i];
				cv::Rect box = cv::Rect(facebox.x0, facebox.y0, facebox.x1 - facebox.x0, facebox.y1 - facebox.y0);
				cv::rectangle(image, box, cv::Scalar(255, 0, 21), 2);
				cv::putText(image, std::to_string(facebox.score), Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1);
			}

			cv::namedWindow("MNN", WINDOW_NORMAL);
			cv::imshow("MNN", image);
			cv::waitKey(1);
		}
	}
	return 0;
}
