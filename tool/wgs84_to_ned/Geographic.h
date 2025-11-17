#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <Eigen/Dense>

class Geographic {
 public:
    GeographicLib::LocalCartesian geographic;
    GeographicLib::Geocentric geocentric;
    void GeographicReset(const Eigen::Vector3d lla);
    Eigen::Vector3d CalculateNed(const Eigen::Vector3d lla);
 private:
    bool init{false};
};

void Geographic::GeographicReset(const Eigen::Vector3d lla) {
    // 创建一个Geodesic和LocalCartesian对象：
    geocentric = GeographicLib::Geocentric::WGS84();
    if (!init) {
        geographic.Reset(lla.x(), lla.y(), lla.z());
        init = true;
    }
}

Eigen::Vector3d Geographic::CalculateNed(const Eigen::Vector3d lla) {
    // 将p2的经纬度转换为p1为原点的ENU坐标系下的坐标x,y,z：
    double x,y,z;
    geographic.Forward(lla.x(), lla.y(), lla.z(), x, y, z);
    return {x, y, z};
}