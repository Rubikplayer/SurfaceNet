
inline
unsigned long randLong(unsigned long lo, unsigned long hi) {return rand()%(hi-lo)+lo;}

inline
float randFloat(float LO, float HI) {return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));}

inline
void colorize_pt(pcl::PointXYZRGB &pt, int r, int g, int b) { pt.r=r; pt.g=g; pt.b=b;}

inline
std::string randID (std::string name){std::stringstream str_ID; str_ID<<name<<"_"<<rand(); return str_ID.str();}

inline
void randPt_inCube (pcl::PointXYZRGB &Pt_offSurf, Eigen::Vector3f &voxel_min, Eigen::Vector3f &voxel_max)
{
    Pt_offSurf.x = randFloat(voxel_min.x() , voxel_max.x());
    Pt_offSurf.y = randFloat(voxel_min.y() , voxel_max.y());
    Pt_offSurf.z = randFloat(voxel_min.z() , voxel_max.z());
}





