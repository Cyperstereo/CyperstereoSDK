#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <Geographic.h>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
    std::string input_path = "gnss.csv";
    if (argc >= 2) {
        input_path = argv[1];
    }

    std::ifstream csv_file(input_path);
    if (!csv_file.is_open()) {
        std::cerr << "failed to open " << input_path << std::endl;
        return -1;
    }

    std::ofstream rtk_output("./rtk.csv", std::ios::trunc);
    if (!rtk_output.is_open()) {
        std::cerr << "failed to open rtk.csv for writing" << std::endl;
        return -1;
    }
    rtk_output.setf(std::ios::fixed, std::ios::floatfield);

    Geographic geographic;
    bool origin_initialized = false;

    std::string line;
    size_t line_number = 0;
    while (std::getline(csv_file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> fields;
        while (std::getline(ss, token, ',')) {
            fields.push_back(token);
        }

        if (fields.size() < 6) {
            std::cerr << "skip line " << line_number << " due to insufficient columns" << std::endl;
            continue;
        }

        double timestamp = 0.0;
        double latitude = 0.0;
        double longitude = 0.0;
        double altitude = 0.0;
        int fix_type = 0;

        try {
            timestamp = std::stod(fields[0]);
            latitude = std::stod(fields[2]);
            longitude = std::stod(fields[3]);
            altitude = std::stod(fields[4]);
            fix_type = std::stoi(fields[5]);
        } catch (const std::exception& e) {
            std::cerr << "skip line " << line_number << " due to conversion failure: " << e.what() << std::endl;
            continue;
        }

        if (fix_type != 4) {
            continue;
        }

        if (!origin_initialized && std::abs(timestamp) < 1e-9) {
            continue;
        }

        Eigen::Vector3d lla{latitude, longitude, 49};
        if (!origin_initialized) {
            geographic.GeographicReset(lla);
            origin_initialized = true;
        }

        Eigen::Vector3d ned = geographic.CalculateNed(lla);
        std::cout << ned.transpose() << std::endl;

        rtk_output.precision(6);
        rtk_output << timestamp << " ";
        rtk_output.precision(5);
        rtk_output << ned.x() << " "
                   << ned.y() << " "
                   << ned.z() << " "
                   << 1 << " "
                   << 0 << " "
                   << 0 << " "
                   << 0
                   << std::endl;
    }

    if (!origin_initialized) {
        std::cerr << "no valid LLA data found in " << input_path << std::endl;
        return -1;
    }

    return 0;
}
