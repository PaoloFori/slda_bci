#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <rosneuro_msgs/NeuroOutput.h> 
#include "slda_bci/utils.hpp" 
#include <string>
#include <vector>


class LoggerNode {
public:
    LoggerNode(ros::NodeHandle& nh) {
        if (!nh.getParam("output_filename", this->output_filename_)) {
            this->output_filename_ = "features_output.csv";
            ROS_WARN("Parameter 'output_filename' doesn't found. Default: %s", this->output_filename_.c_str());
        }

        if (!nh.getParam("paradigm", this->paradigm_)) {
            this->paradigm_ = "mi";
            ROS_WARN("Parameter 'paradigm' doesn't found. Default: %s", this->paradigm_.c_str());
        }

        std::string topic = "/" + this->paradigm_ + "/neuroprediction/raw";
        sub_ = nh.subscribe(topic, 10, &LoggerNode::callback, this);
    }

    ~LoggerNode() {
        if (collected_rows_.empty()) {
            ROS_WARN("No data received.");
            return;
        }

        int n_samples = collected_rows_.size();
        int n_classes = collected_rows_[0].cols();

        ROS_INFO("Received %d samples. Creation final matrix [samples x classes]: (%d x %d)...",
                 n_samples, n_samples, n_classes);

        Eigen::MatrixXf final_matrix(n_samples, n_classes);

        for (int i = 0; i < n_samples; ++i) {
            final_matrix.row(i) = collected_rows_[i];
        }

        writeCSV<float>(output_filename_, final_matrix);
    }


    void callback(const rosneuro_msgs::NeuroOutput::ConstPtr& msg) {

        const std::vector<float>& prob_vector = msg->softpredict.data;

        Eigen::Map<const Eigen::RowVectorXf> prob_row(
            prob_vector.data(), 
            prob_vector.size()
        );

        collected_rows_.push_back(prob_row);

        ROS_INFO("ID_seq: %d.", msg->neuroheader.seq);
    }

private:
    ros::Subscriber sub_;
    std::string output_filename_;
    std::string paradigm_;
    
    std::vector<Eigen::RowVectorXf> collected_rows_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_subscriber");
    ros::NodeHandle nh("~"); 

    LoggerNode logger(nh);

    ros::spin(); 

    return 0; 
}