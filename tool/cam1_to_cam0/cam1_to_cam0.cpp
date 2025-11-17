#include <iostream>
#include <fstream>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <map>
#include <Eigen/Dense>

using namespace std;


int main(int argc, char** argv)
{   
    Eigen::Matrix3d R;
    R << 0.99984656, -0.01059921, 0.01394665, 
          0.01053888, 0.99993482, 0.00439234,
          -0.01399229, -0.00424469, 0.99989309;
    Eigen::Vector3d t;
    t << -0.15029288, -0.00048704, -0.00046303;
    
    std::cout << R.transpose() << std::endl;
    std::cout << - R.transpose() * t << std::endl;
    return 0;

}
