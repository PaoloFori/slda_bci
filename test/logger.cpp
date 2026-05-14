#include <ros/ros.h>
#include <rosneuro_msgs/NeuroOutput.h>
#include <signal.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>

static bool g_shutdown = false;
static void sigint_handler(int) { g_shutdown = true; ros::shutdown(); }

class SldaLogger {
public:
    SldaLogger(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        pnh.param<std::string>("output_filename", output_filename_, "/tmp/slda_output.csv");
        pnh.param<std::string>("paradigm", paradigm_, "mi");

        // Create output directory if it doesn't exist
        auto slash = output_filename_.rfind('/');
        if (slash != std::string::npos) {
            std::string dir = output_filename_.substr(0, slash);
            std::system(("mkdir -p " + dir).c_str());
        }

        std::string topic = "/" + paradigm_ + "/neuroprediction/raw";
        sub_ = nh.subscribe(topic, 200, &SldaLogger::callback, this);
        ROS_INFO("[SldaLogger] Subscribing to %s → %s", topic.c_str(), output_filename_.c_str());
    }

    ~SldaLogger() { save(); }

private:
    void callback(const rosneuro_msgs::NeuroOutput::ConstPtr& msg) {
        if (first_seq_ < 0) {
            first_seq_ = (int)msg->neuroheader.seq;
            n_classes_ = (int)msg->softpredict.data.size();
            ROS_INFO("[SldaLogger] first_seq=%d  nclasses=%d", first_seq_, n_classes_);
        }
        flat_data_.insert(flat_data_.end(),
                          msg->softpredict.data.begin(),
                          msg->softpredict.data.end());
    }

    void save() {
        if (flat_data_.empty()) { ROS_WARN("[SldaLogger] No data to save."); return; }

        int total_frames = (int)flat_data_.size() / n_classes_;

        std::ofstream f(output_filename_);
        if (!f) { ROS_ERROR("[SldaLogger] Cannot open %s", output_filename_.c_str()); return; }
        f << std::setprecision(9);
        for (int r = 0; r < total_frames; r++) {
            for (int c = 0; c < n_classes_; c++) {
                if (c > 0) f << ",";
                f << flat_data_[r * n_classes_ + c];
            }
            f << "\n";
        }
        ROS_INFO("[SldaLogger] Saved %d × %d to %s", total_frames, n_classes_, output_filename_.c_str());

        std::string seq_file = output_filename_.substr(0, output_filename_.rfind(".csv")) + "_first_seq.txt";
        std::ofstream fs(seq_file);
        fs << first_seq_ << "\n";
        ROS_INFO("[SldaLogger] Saved first_seq=%d to %s", first_seq_, seq_file.c_str());
    }

    ros::Subscriber sub_;
    std::string output_filename_;
    std::string paradigm_;
    std::vector<float> flat_data_;
    int n_classes_ = 0;
    int first_seq_ = -1;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "slda_logger", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    SldaLogger logger(nh, pnh);
    ros::spin();
    return 0;
}
