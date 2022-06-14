#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>       // std::vector
#include <cstdlib>

using namespace cv;
using namespace std;

//-----------------Medir tiempos de ejecucion----------
namespace cpp_secrets{
///Runnable: A class which has a valid and public default ctor and a "run()" function.
///BenchmarkingTimer tests the "run()" function of Runnable
///num_run_cycles: It is the number of times run() needs to be run for a single test.
///One Runnable object is used for a single test.
///Note: if the run() function is statefull then it can only be run once for an object in order
///to get meaningful results.
///num_tests: It is the number of tests that need to be run.
    template <typename Runnable, int num_run_cycles = 1000000, int num_tests = 10>

        struct BenchmarkingTimer{
            /// runs the run() function of the Runnable object and captures timestamps around each test
            void run(){
                for(int i = 0; i < num_tests; i++){
                    Runnable runnable_object_{};
                    Timer t{intervals_[i].first, intervals_[i].second};
                    for(int i = 0; i < num_run_cycles; i++){
                        runnable_object_.run();
                    }
                }
            }

            ///utility function to print durations of all tests
            std::string durations() const{
                std::stringstream ss;
                int i{1};
                for(const auto& interval: intervals_){
                    ss << "Test-" << i++  << " duration = " << (interval.second - interval.first) * 0.001 << " ms" << std::endl;
                }
                return ss.str();
            }

            ///utility function to print average duration of all tests
            double average_duration(){
                auto duration_sum{0.0};
                for(const auto& interval: intervals_){
                    duration_sum += (interval.second - interval.first) * 0.001;
                }
                if (num_tests) return (duration_sum/num_tests);
                return 0;
            }

            private:
            std::array<std::pair<double, double>, num_tests> intervals_{};

            struct Timer{
                Timer(double& start, double& finish):finish_(finish) { start = now(); }
                ~Timer() { finish_ = now(); }

                private:
                double& finish_;
                double now(){
                         ///utility function to return current time in microseconds since epoch
                    return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
                }
            };

        };
}

///sample class which has a statefull run().
//run() function is stateful because it is not meaningful to sort a sorted array.
//that's why num_run_cycles = 1 in this case.
struct randomly_sorted{
    randomly_sorted(){
        srand(time(0));
        for(int i=0;i<1000000;i++){
            arr_.emplace_back(rand());
                  // making a vector filled with random elements
        }
    }

    void run(){
        sort(arr_.begin(), arr_.end(), std::less<int>());
    }
    private:
    std::vector<int>arr_;
};

//-----------------Funcion clahe-----------------------
Mat claheGO(Mat src,int _step = 8)
{
    Mat CLAHE_GO = src.clone();
    int block = _step;//pblock
    int width = src.cols;
    int height= src.rows;
    int width_block = width/block;
    int height_block = height/block;

    int tmp2[8*8][256] ={0};
    float C2[8*8][256] = {0.0};

    int total = width_block * height_block;
    //#pragma omp parallel for collapse (2)
    for (int i=0;i<block;i++)
    {
        for (int j=0;j<block;j++)
        {
            int start_x = i*width_block;
            int end_x = start_x + width_block;
            int start_y = j*height_block;
            int end_y = start_y + height_block;
            int num = i+block*j;

            for(int ii = start_x ; ii < end_x ; ii++)
            {
                for(int jj = start_y ; jj < end_y ; jj++)
                {
                    int index =src.at<uchar>(jj,ii);
                    tmp2[num][index]++;
                }
            }

            int average = width_block * height_block / 255;

            int LIMIT = 40 * average;
            int steal = 0;
            for(int k = 0 ; k < 256 ; k++)
            {
                if(tmp2[num][k] > LIMIT){
                    steal += tmp2[num][k] - LIMIT;
                    tmp2[num][k] = LIMIT;
                }
            }
            int bonus = steal/256;
            //hand out the steals averagely
            for(int k = 0 ; k < 256 ; k++)
            {
                tmp2[num][k] += bonus;
            }

            for(int k = 0 ; k < 256 ; k++)
            {
                if( k == 0)
                    C2[num][k] = 1.0f * tmp2[num][k] / total;
                else
                    C2[num][k] = C2[num][k-1] + 1.0f * tmp2[num][k] / total;
            }
        }
    }

    for(int  i = 0 ; i < width; i++)
    {
        for(int j = 0 ; j < height; j++)
        {
            //four coners
            if(i <= width_block/2 && j <= height_block/2)
            {
                int num = 0;
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);
            }else if(i <= width_block/2 && j >= ((block-1)*height_block + height_block/2)){
                int num = block*(block-1);
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);
            }else if(i >= ((block-1)*width_block+width_block/2) && j <= height_block/2){
                int num = block-1;
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);
            }else if(i >= ((block-1)*width_block+width_block/2) && j >= ((block-1)*height_block + height_block/2)){
                int num = block*block-1;
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);
            }
            //four edges except coners
            else if( i <= width_block/2 )
            {

                int num_i = 0;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + block;
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                float q = 1-p;
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);
            }else if( i >= ((block-1)*width_block+width_block/2)){

                int num_i = block-1;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + block;
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                float q = 1-p;
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);
            }else if( j <= height_block/2 ){

                int num_i = (i - width_block/2)/width_block;
                int num_j = 0;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float q = 1-p;
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);
            }else if( j >= ((block-1)*height_block + height_block/2) ){

                int num_i = (i - width_block/2)/width_block;
                int num_j = block-1;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float q = 1-p;
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);
            }

            else{
                int num_i = (i - width_block/2)/width_block;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                int num3 = num1 + block;
                int num4 = num2 + block;
                float u = (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float v = (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                CLAHE_GO.at<uchar>(j,i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j,i)] +
                    (1-v)*(1-u)*C2[num1][CLAHE_GO.at<uchar>(j,i)] +
                    u*(1-v)*C2[num2][CLAHE_GO.at<uchar>(j,i)] +
                    v*(1-u)*C2[num3][CLAHE_GO.at<uchar>(j,i)]) * 255);
            }
            //smooth
            CLAHE_GO.at<uchar>(j,i) = CLAHE_GO.at<uchar>(j,i) + (CLAHE_GO.at<uchar>(j,i) << 8) + (CLAHE_GO.at<uchar>(j,i) << 16);
        }
    }
  return CLAHE_GO;
}


int main()
{

Mat src = cv::imread("/home/boris/Documentos/claheCV2/pel.jpg");
Mat original;
cvtColor(src, original, COLOR_BGR2GRAY);
//
// Extract the L channel

std::vector<cv::Mat> channels (3);
split(src, channels);

channels.at(0) = claheGO(channels.at(0));
channels.at(1) = claheGO(channels.at(1));
channels.at(2) = claheGO(channels.at(2));
merge(channels, src);
Mat clahe_image(src.size(), CV_8UC1);
for (size_t i = 0; i < src.rows; i++)
    {
        for (size_t j = 0; j < src.cols; j++)
        {
            clahe_image.at<uchar>(i, j) = std::max(
                src.at<cv::Vec3b>(i, j)[0],
                std::max(
                    src.at<cv::Vec3b>(i, j)[1],
                    src.at<cv::Vec3b>(i, j)[2]
                )
            );
        }
    }
imshow("Imagen Original", original);
imshow("Imagen ClAHE", clahe_image);
cv::waitKey(0);

cpp_secrets::BenchmarkingTimer<randomly_sorted, 1, 10> test; // randomly_sorted structure run function is run 10 time and average output is given.
        test.run();
        std::cout << test.durations() << std::endl; // outputs the duration of every test.
        std::cout << "average duration = " << test.average_duration() << " ms" << std::endl;
//destroyALLwindows();
return 0;
}
