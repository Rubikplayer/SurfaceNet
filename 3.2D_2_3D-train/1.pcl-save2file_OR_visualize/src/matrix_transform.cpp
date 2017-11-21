#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/copy_point.h>

#include <pcl/octree/octree.h>
#include <pcl/registration/distances.h> //distance
#include <pcl/surface/texture_mapping.h>
#include <algorithm>
#include <sstream>

inline
unsigned long randLong(unsigned long lo, unsigned long hi) {return rand()%(hi-lo)+lo;}
inline
float randFloat(float LO, float HI) {return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));}
inline
void randPt_inCube (pcl::PointXYZRGB &Pt_offSurf, Eigen::Vector3f &voxel_min, Eigen::Vector3f &voxel_max)
{
    Pt_offSurf.x = randFloat(voxel_min.x() , voxel_max.x());
    Pt_offSurf.y = randFloat(voxel_min.y() , voxel_max.y());
    Pt_offSurf.z = randFloat(voxel_min.z() , voxel_max.z());
}
void randPt_inCube (pcl::PointXYZRGB &Pt_offSurf, double& min_x_arg, double& min_y_arg, double& min_z_arg,
                    double& max_x_arg, double& max_y_arg, double& max_z_arg)
{
    Pt_offSurf.x = randFloat(min_x_arg , max_x_arg);
    Pt_offSurf.y = randFloat(min_y_arg , max_y_arg);
    Pt_offSurf.z = randFloat(min_z_arg , max_z_arg);
}


// This function displays the help
void
showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
  std::cout << "-h:  Show this help." << std::endl;
}


bool
GeneratePt_offSurface(const int &depth, const unsigned long &iterator_max,
                      pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree,
                      pcl::PointXYZRGB &Pt_offSurf,
                      const float &thresh_offsurf_min,
                      const float &thresh_offsurf_max)
{
    bool Pt_offSurf_valid = false;
    int apprNN_Indx;
    float apprNN_dist;
    Eigen::Vector3f voxel_min, voxel_max;
    std::vector<int> pointIdxVec;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Iterator node_itr;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Iterator node_itr_end = octree.end();
    // reinitialize a random access point of the node_itrerator. range(1,iterator_counter)
    while(unsigned long rand_node_starter = rand()%iterator_max+1)
    {
        unsigned long node_counter = 0;
        for (node_itr = octree.begin(depth); node_itr!= node_itr_end; ++node_itr)
        {
            node_counter++;
            // if the current depth is == depth, and access to the accPTrand-th iteration
            if(node_itr.getCurrentOctreeDepth() == depth &&
                    node_counter == rand_node_starter)
            {
                rand_node_starter++; // make sure that from the rand_acc_pt, later iterations will run into here.

                // if still cannot generate proper Pt_offSurfom, change to another big cube/branch of node_itr
                for(int tmp=0; tmp<10; tmp++)
                {
                    octree.getVoxelBounds(node_itr, voxel_min, voxel_max);
                    randPt_inCube(Pt_offSurf, voxel_min, voxel_max);
                    // if the voxel containing the Pt_offSurf is empt_imgy, use this Pt_offSurf; otherwise, rand again
                    if (!octree.voxelSearch (Pt_offSurf, pointIdxVec)) //"true" if leaf node exist
                    {
                        octree.approxNearestSearch(Pt_offSurf, apprNN_Indx, apprNN_dist);
                        if(apprNN_dist > thresh_offsurf_min && apprNN_dist < thresh_offsurf_max) //approximate Nearest Neighbor is larger than threshold
                        {
                            //printf("apprNN_dist: %f < %f < %f \n", thresh_offsurf_min ,apprNN_dist, thresh_offsurf_max);
                            Pt_offSurf_valid = true; // flag
                            return true;
                        }
                    }
                }
            }
        }
    }
}



bool
GeneratePt_offSurface_simple(pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree,
                      pcl::PointXYZRGB &Pt_offSurf,
                      const float &thresh_offsurf_min,
                      const float &thresh_offsurf_max)
{
    bool Pt_offSurf_valid = false;
    int apprNN_Indx;
    float apprNN_dist;
    std::vector<int> pointIdxVec;

    double min_x, min_y, min_z, max_x, max_y, max_z;
    octree.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
//    std::cout<<" min_x:"<<min_x<<" max_x:"<<max_x<<" min_y:"<<min_y<<" max_y:"<<max_y<<" min_z:"<<min_z<<" max_z:"<<max_z<<"\n";
    // reinitialize a random access point of the node_itrerator. range(1,iterator_counter)
    while(1)
    {
        randPt_inCube(Pt_offSurf, min_x, min_y, min_z, max_x, max_y, max_z);
        // if the voxel containing the Pt_offSurf is empt_imgy, use this Pt_offSurf; otherwise, rand again
        if (!octree.voxelSearch (Pt_offSurf, pointIdxVec)) //"true" if leaf node exist
        {
            octree.approxNearestSearch(Pt_offSurf, apprNN_Indx, apprNN_dist);
            if(apprNN_dist > thresh_offsurf_min && apprNN_dist < thresh_offsurf_max) //approximate Nearest Neighbor is larger than threshold
            {
                //printf("apprNN_dist: %f < %f < %f \n", thresh_offsurf_min ,apprNN_dist, thresh_offsurf_max);
                Pt_offSurf_valid = true; // flag
                return true;
            }
        }
    }
}

int
read_cameraPO(const int &viewIndx, Eigen::Matrix4d &matA)
{
    // stringstream --> const string --> const char*.  fuck!!!
    std::stringstream fileName;
    fileName<<"/hdd1t/dataset/miniDataset/pos/pos_0"<<std::setw(2)<<std::setfill('0')<<viewIndx<<".txt";
    const std::string fileName_tmp = fileName.str();
    std::ifstream po_file(fileName_tmp.c_str());
    //po_file.open();//"../../miniDataset/pos/pos_012.txt");//
    double po_elem [4*4];
    if (!po_file) {
        std::cout << "Cannot open file:" << fileName_tmp << "\n";
        return -1;
    }
    // assigne elemnt of array
    for (int j = 0; j < 12; j++) {
        po_file >> po_elem[j];
        ////printf("*%f* ", po_elem[j]);
    }
    po_elem[12]=0; po_elem[13]=0; po_elem[14]=0; po_elem[15]=1;

    matA = Eigen::Map<Eigen::MatrixXd>(po_elem,4,4).transpose();
    //std::cout<<"\n"<<matA<<std::endl;
    po_file.close();
}


int
read_cameraT(const int &viewIndx, pcl::PointXYZRGB &matT)
{
    // stringstream --> const string --> const char*. ???
    std::stringstream fileName;
    fileName<<"/hdd1t/dataset/miniDataset/cameraT/T"<<std::setw(2)<<std::setfill('0')<<viewIndx<<".txt";
    const std::string fileName_tmp = fileName.str();
    std::ifstream po_file(fileName_tmp.c_str());
    //po_file.open();//"../../miniDataset/pos/pos_012.txt");//
    double po_elem [3];
    if (!po_file) {
        std::cout << "Cannot open file." << fileName_tmp << "\n";
        return -1;
    }
    // assigne elemnt of array
    for (int j = 0; j < 3; j++) {
        po_file >> po_elem[j];
        ////printf("*%f* ", po_elem[j]);
    }
    matT.x = po_elem[0];
    matT.y = po_elem[1];
    matT.z = po_elem[2];


    //std::cout<<"\n"<<matA<<std::endl;
    po_file.close();
}

bool
Point_isVisible( const pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree,
                 const pcl::PointXYZRGB &source_point,
                 const pcl::PointXYZRGB &cameraT,
                 const float &thresh_occlusion)
{
    //from the source_pt cast a ray to transformed_pt, if the 1st intersected voxel don't include the point --> invisible
    // getIntersectedVoxelIndices returns the point indices. those points are in the intersected voxels.
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::AlignedPointTVector voxel_center_list;
    int NO_intersectedVoxels = octree.getIntersectedVoxelCenters(source_point.getVector3fMap(),
                                                                 cameraT.getVector3fMap()-source_point.getVector3fMap(),
                                                                 voxel_center_list, 0);// stop at the first intersected voxel
    // if there are more/less than 1 voxel on the casted ray, it means that this source point is invisible.
    // but for an invisible point behind surface, the NO_intersectedVoxels == 1
    // so that need to further check whether the intersectedVoxel include the source_pt or not, use 'isVoxelOccupiedAtPoint'
    //printf("%d,%d;\t", NO_intersectedVoxels ,octree.isVoxelOccupiedAtPoint(source_point));
    if (NO_intersectedVoxels == 0)
        return true;

    //    float d;
    //    for (int i=0; i<NO_intersectedVoxels; i++)
    //    {
    //        d = pcl::distances::l2Sqr(voxel_center_list[i].getVector4fMap(),
    //                    source_point.getVector4fMap());
    //        printf("d2farestCenter: %f\t", d);
    //    }

    float d2farestCenter = pcl::distances::l2Sqr(voxel_center_list[NO_intersectedVoxels-1].getVector4fMap(),
            source_point.getVector4fMap());
    return d2farestCenter < thresh_occlusion;
}


void
Point_Perspective_Projection(const pcl::PointXYZRGB &source_point,
                             pcl::PointXYZRGB &transformed_point,
                             const Eigen::Matrix4d &transform,
                             double &ptdepth)
{
    transformed_point = pcl::transformPoint (source_point, (Eigen::Affine3d)transform);
    ptdepth = transformed_point.z;
    transformed_point.x /= transformed_point.z;
    transformed_point.y /= transformed_point.z;
    transformed_point.z = 1;
}


bool
Point_isinImgScope(const pcl::PointXYZRGB &source_point,
                   const Eigen::Matrix4d &transform,
                   int& pt_imgx, int& pt_imgy, double& ptdepth,
                   const float &x=1600, const float &y=1200)
{
    int safe_margin = 20 ; // cut lice around boundary
    pcl::PointXYZRGB transformed_point;
    Point_Perspective_Projection(source_point,transformed_point, transform, ptdepth);
    pt_imgx = int(transformed_point.x);
    pt_imgy = int(transformed_point.y);
    return transformed_point.x >= (1+safe_margin) && transformed_point.x <= (x-safe_margin) &&
            transformed_point.y >= (1+safe_margin) && transformed_point.y <= (y-safe_margin);
}



bool
Voxel_isVisible (pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree,
                 const pcl::PointXYZRGB &Pt_onSurf,
                 const pcl::PointXYZRGB &cameraT,
                 const float &thresh_occlusion)
{
    std::vector<int> pointIdxVec;
    // if the Pt_onSurf is not in any voxel, check point_isvisible
    if (!octree.voxelSearch(Pt_onSurf, pointIdxVec)) // cannot use the argument of const octree
    {
        //printf("voxel_isvisible: a point NOT on surface");
        return Point_isVisible(octree,Pt_onSurf, cameraT, thresh_occlusion);
    }
    // else: iterate all the points in this voxel,
    // if there is one visible point, stop iteration, return true;
    else
    {
        //printf("voxel_isvisible: a point ON surface");
        for (size_t i = 0; i < pointIdxVec.size (); ++i)
        {
            // source_point is reference of the point in the source_cloud, because
            // later on, the color of this point will be changed.
            // if only read some property of this point, don't need '&'
            pcl::PointXYZRGB pt_tmp = octree.getInputCloud()->points[pointIdxVec[i]];

            if (Point_isVisible(octree, pt_tmp, cameraT, thresh_occlusion))
            {
                //continue; // used to colorize all the points in the voxel
                //break; // can be used to check whether a voxel is visible or not
                return true;
            }
        }
        return false;
    }
}


// This is the main function
int
main (int argc, char** argv)
{

//  // Show help
//  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
//    showHelp (argv[0]);
//    return 0;
//  }

//  // Fetch point cloud filename in arguments | Works with PCD and PLY files
//  std::vector<int> filenames;
//  bool file_is_pcd = false;

//  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
//  if (filenames.size () != 1)  {
//    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

//    if (filenames.size () != 1) {
//      showHelp (argv[0]);
//      return -1;
//    } else {
//      file_is_pcd = true;
//    }
//  }



//  // Load file | Works with PCD and PLY files
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

//  if (file_is_pcd) {
//    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
//      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
//      showHelp (argv[0]);
//      return -1;
//    }
//  } else {
//    if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
//      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
//      showHelp (argv[0]);
//      return -1;
//    }
//  }

  int pc_Indx = 1;
  std::string mode = "hard";
  if (argc == 2)
  {
      std::istringstream iss( argv[1] );
      if (!(iss >> pc_Indx))
          printf("error argv[1]");
  }

  std::stringstream pc_fileName;
  pc_fileName<<"/home/mengqi/dataset/MVS/Points/stl/stl"<<std::setw(3)<<std::setfill('0')<<pc_Indx<<"_total.ply";

  std::stringstream output_fileName;
  output_fileName<<"/home/mengqi/dataset/MVS/lasagne/samplesVoxelVolume/pcl_txt_50x50x50_2D_2_3D/output_stl_"<<std::setw(3)<<std::setfill('0')<<pc_Indx<<".txt";
  const std::string output_fileName_tmp = output_fileName.str();
  std::ofstream output_file;

  srand (static_cast <unsigned> (time(0)));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

  if (pcl::io::loadPLYFile (pc_fileName.str(), *source_cloud) < 0)  {
      std::cout << "Error loading point cloud ";
      return -1;
  }

  /////////////parameter
  bool Visualization_ON = false;


  pcl::visualization::PCLVisualizer *viewer;
  //printf("\n width %d;\t height %d\n", source_cloud->width, source_cloud->height); //test

  int NO_iteration = 100; // NO of cubes to be sampled
  int NO_offSurfCubes = 0; // NO_iteration/2; // or 0, how many empty cubes are wanted.
  int grids_d = 50; // the cube has dim: dxdxd
  float resolution = 0.4f;

  if(Visualization_ON)
  {
    viewer = new pcl::visualization::PCLVisualizer("Matrix transformation example");
    NO_iteration = 2;
    NO_offSurfCubes = 50;
    grids_d = 10;
  }
  else{
    output_file.open(output_fileName_tmp.c_str(), ios::trunc);
  }

  float cube_d = grids_d*resolution;
  //const char* output_file_name = "output.txt";


  pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (resolution);
  octree.setInputCloud (source_cloud);
  octree.addPointsFromInputCloud ();


  std::vector<int> pointIdxVec;
  pcl::PointXYZRGB searchPoint;
  pcl::PointXYZRGB currentPt;
  std::stringstream oneLineStr("");
  std::ostringstream sphere_name;
  long searchPtIdx;
  float cube_min_x, cube_min_y, cube_min_z, cube_max_x, cube_max_y, cube_max_z;


    int *viewIdx;
    Eigen::Matrix4d transform_M ;//= Eigen::Matrix4d::Identity();
    pcl::PointXYZRGB  cameraT;
    int pt_imgx, pt_imgy; // the img postiion of the pt
    double ptdepth; // used to save the depth of current point
    float thresh_occlusion = 5.0f; // used to check point's visibility
    int img_x = 1600;
    int img_y = 1200;

    int viewNO = 0;
    int viewIdx_generation[64];
    for (int i=0;i<64;i++)
        viewIdx_generation[i] = i+1;

    viewIdx = viewIdx_generation;
    viewNO = sizeof(viewIdx_generation)/sizeof(viewIdx_generation[0]);
 

  for(int c=0; c<NO_iteration; c++)
  {
      // randly select a pt, crop a cube around it
      searchPtIdx = randLong(1,source_cloud->width);
      searchPoint = source_cloud->points[searchPtIdx];

      cube_min_x = searchPoint.x -0.5*cube_d; //+ randFloat(-0.5*cube_d,0);
      cube_min_y = searchPoint.y -0.5*cube_d; //+ randFloat(-0.5*cube_d,0);
      cube_min_z = searchPoint.z -0.5*cube_d; //+ randFloat(-0.5*cube_d,0);
      cube_max_x = cube_min_x + cube_d;
      cube_max_y = cube_min_y + cube_d;
      cube_max_z = cube_min_z + cube_d;

      currentPt.x = cube_min_x;
      currentPt.y = cube_min_y;
      currentPt.z = cube_min_z;

      oneLineStr<<cube_min_x<<" "<<cube_min_y<<" "<<cube_min_z<<" "<<resolution<<",";

      // this can be used to random select the order of xyz increament. But this can be much each to be performed in Python.
      float *xyz[3] = {&currentPt.x,&currentPt.y,&currentPt.z};


      for(int xi=0;xi<grids_d;xi++, *xyz[0] += resolution)
      {
          for(int yi=0;yi<grids_d;yi++, *xyz[1] += resolution)
          {
              for(int zi=0;zi<grids_d;zi++, *xyz[2] += resolution)
              {
                  if (Visualization_ON)
                  {
                      //// visualize the voxels using sphere, better to use small grids_d value
                      std::cout << "\t current testing on: " << xi<<'\t'<<currentPt.x<<'\t'<< yi<<'\t'<<currentPt.y << '\t' <<zi<<'\t'<<currentPt.z<< "\n";
                      sphere_name<<"sphere"<<xi<<yi<<zi;
                      viewer->addSphere (currentPt, resolution/2, 0.0, 0.0, 0.8, sphere_name.str());
                  }
                  // there exist pts in this voxel
                  if (octree.voxelSearch(currentPt, pointIdxVec))
                  {
                      //std::cout << "\t WRITE A LINE: \n" ;//<< oneLineStr.str() << "\n";
                      oneLineStr<< xi<<" "<<yi<<" "<<zi<<" "<< pointIdxVec.size() <<",";
                      pointIdxVec.clear(); //If don't clear, the voxelSeaerch method will append the result to the vector~
                  }
              }
              currentPt.z = cube_min_z;
          }
          currentPt.y = cube_min_y;
      }
      oneLineStr << ";";

      for(int indx=0; indx<viewNO; indx++)
      {
        read_cameraPO(viewIdx[indx], transform_M);
        read_cameraT(viewIdx[indx],cameraT);
        bool in_imgScope = Point_isinImgScope(searchPoint,transform_M, pt_imgx, pt_imgy, ptdepth, img_x, img_y);
        bool voxel_visible = Voxel_isVisible(octree, searchPoint,cameraT, thresh_occlusion);
        // oneLineStr<< viewIdx[indx]<<" "<<pt_imgx<<" "<<pt_imgy<<" "<< ptdepth << " "<<int(voxel_visible)<<";";
        oneLineStr<< viewIdx[indx]<<" "<<int(in_imgScope & voxel_visible)<<",";
      }
      // finished one cube
      oneLineStr << "\n";
  }

  pcl::PointXYZRGB Pt_offSurf;
  float thresh_offsurf_min = cube_d * 2; // used to check whether pt is off surface
  float thresh_offsurf_max = 1000.0f; // used to check whether pt is off surface

//  std::cout << "\t octree.getTreeDepth(): " <<octree.getTreeDepth()<<"\n";
  for(int j=0; j<NO_offSurfCubes; j++)
  {
    GeneratePt_offSurface_simple(octree,Pt_offSurf, thresh_offsurf_min, thresh_offsurf_max);

    cube_min_x = Pt_offSurf.x ;
    cube_min_y = Pt_offSurf.y ;
    cube_min_z = Pt_offSurf.z ;
    oneLineStr<<cube_min_x<<" "<<cube_min_y<<" "<<cube_min_z<<" "<<resolution<<",;"; //finish one empty cube
    if (Visualization_ON)
    {
        //// visualize the voxels using sphere, better to use small grids_d value
        std::cout << "\t current testing on: " <<Pt_offSurf.x<<'\t'<<Pt_offSurf.y <<'\t'<<Pt_offSurf.z<< "\n";
        sphere_name<<"sphere"<<j;
        viewer->addSphere (Pt_offSurf, cube_d/2, 0.3, 0.8, 0.1, sphere_name.str());
    }

      for(int indx=0; indx<viewNO; indx++)
      {
        read_cameraPO(viewIdx[indx], transform_M);
        read_cameraT(viewIdx[indx],cameraT);
        bool in_imgScope = Point_isinImgScope(Pt_offSurf,transform_M, pt_imgx, pt_imgy, ptdepth, img_x, img_y);
        bool voxel_visible = Voxel_isVisible(octree, Pt_offSurf,cameraT, thresh_occlusion);
        // oneLineStr<< viewIdx[indx]<<" "<<pt_imgx<<" "<<pt_imgy<<" "<< ptdepth << " "<<int(voxel_visible)<<";";
        oneLineStr<< viewIdx[indx]<<" "<<int(in_imgScope & voxel_visible)<<",";
      }
      oneLineStr << "\n";
  }

  if (Visualization_ON)
  {
    //// Visualization
    //  viewer->addCube(cube_min_x, cube_max_x, cube_min_y, cube_max_y, cube_min_z, cube_max_z);
    //  viewer->addSphere (searchPoint, 2, 0.0, 0.0, 0.8, "sphere_orig");
    //  if (octree.voxelSearch (searchPoint, pointIdxVec))
    //  {
    //    std::cout << "Neighbors within voxel search at (" << searchPoint.x
    //     << " " << searchPoint.y
    //     << " " << searchPoint.z << ")"
    //     << std::endl;

    //    for (size_t i = 0; i < pointIdxVec.size (); ++i)
    //    {
    //        std::cout << "    " << source_cloud->points[pointIdxVec[i]].x
    //           << " " << source_cloud->points[pointIdxVec[i]].y
    //           << " " << source_cloud->points[pointIdxVec[i]].z << std::endl;
    //        source_cloud->points[pointIdxVec[i]].r=0;
    //        source_cloud->points[pointIdxVec[i]].g=255;
    //        source_cloud->points[pointIdxVec[i]].b=0;
    //    }
    //  }


      // Define R,G,B colors for the point cloud
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> source_cloud_color_handler(source_cloud);//, 255, 255, 255);
      // add the point clouds to the viewer and pass the color handler
      viewer->addPointCloud<pcl::PointXYZRGB> (source_cloud, source_cloud_color_handler, "original_cloud");

      viewer->addCoordinateSystem (100.0, "cloud", 0);
      viewer->setBackgroundColor(0.15, 0.15, 0.15, 0); // Setting background to a dark grey
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");

      while (!viewer->wasStopped ()) { // Display the visualiser until 'q' key is pressed
        viewer->spinOnce ();
      }
  }
  else{
      output_file << oneLineStr.str(); // << std::endl;
      output_file.close();
  }

  return 0;
}
