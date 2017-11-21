#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/surface/texture_mapping.h>
#include <pcl/registration/distances.h> //distance


#include <algorithm>
#include <sstream>
#include <stdlib.h>     /* srand, rand */
#include <cstdlib> /*RAND_MAX*/
#include <ctime> /*time*/
#include <fstream> /*file*/
#include "matrix_transform.h"



// This function displays the help

void
showHelp(char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
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

void
Cloud_Perspective_Projection(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud,
                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr &transformed_cloud,
                             const Eigen::Matrix4d &transform)
{
    pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform);
    for (size_t i = 0; i < transformed_cloud->points.size(); ++i)
    {
        transformed_cloud->points[i].x /= transformed_cloud->points[i].z;
        transformed_cloud->points[i].y /= transformed_cloud->points[i].z;
        transformed_cloud->points[i].z = 1;
    }
}

bool
Point_isVisible_drawCubes( pcl::visualization::PCLVisualizer &viewer,
                           const pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree,
                           const pcl::PointXYZRGB &source_point,
                           const pcl::PointXYZRGB &cameraT)
{
    //from the source_pt cast a ray to transformed_pt, if the 1st intersected voxel don't include the point --> invisible
    // getIntersectedVoxelIndices returns the point indices. those points are in the intersected voxels.
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::AlignedPointTVector voxel_center_list;
    int NO_intersectedVoxels = octree.getIntersectedVoxelCenters(source_point.getVector3fMap(),
                                                                 cameraT.getVector3fMap() - source_point.getVector3fMap(),
                                                                 voxel_center_list, 0);// stop at the first intersected voxel
    // if there are more/less than 1 voxel on the casted ray, it means that this source point is invisible.
    // but for an invisible point behind surface, the NO_intersectedVoxels == 1
    // so that need to further check whether the intersectedVoxel include the source_pt or not, use 'isVoxelOccupiedAtPoint'
    printf("NO_intersectedVoxel: %d,%d;\t", NO_intersectedVoxels ,octree.isVoxelOccupiedAtPoint(source_point));

    float sidelen = octree.getResolution();
    float halph=sidelen/2;
    pcl::PointXYZRGB ctPt;
    for (int i=0; i<NO_intersectedVoxels; i++)
    {
        ctPt = voxel_center_list[i];
        viewer.addCube(ctPt.x-halph, ctPt.x+halph, ctPt.y-halph, ctPt.y+halph,
                       ctPt.z-halph, ctPt.z+halph, 1.0, 1.0, 1.0, randID("cube"));
        printf("sidelen: %f; distance: %f\n", sidelen, pcl::distances::l2Sqr(voxel_center_list[i].getVector4fMap(), source_point.getVector4fMap()));
    }
    return NO_intersectedVoxels == 1 && octree.isVoxelOccupiedAtPoint(source_point);
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


// This is the main function
int
main (int argc, char** argv)
{
    int pc_Indx = 1;
    std::string mode = "hard";
    if (argc >= 2)
    {
        std::istringstream iss( argv[1] );
        if (!(iss >> pc_Indx))
            printf("error argv[1]");

        if (argc == 3 )
        {
            mode = argv[2];
            if (!(mode.compare("hard")==0 || mode.compare("easy")==0))
                mode = "hard";
        }
    }

    std::cout<<"current pc_Indx: "<<pc_Indx<<"\t current mode: "<<mode<<std::endl;
    //******* parameters
    int visualization_ON = 0;
    int NO_onSurfPt = 5000;
    int NO_offSurfPt = 10000; // most of them are invalid
    if (visualization_ON)
    {
        NO_onSurfPt = 50000;
        NO_offSurfPt = 1000;
    }
    int img_x = 1600;
    int img_y = 1200;

    float thresh_occlusion = 5.0f; // used to check point's visibility
    //float thresh_offsurf = 200.0f; // used to check whether pt is off surface

    float thresh_offsurf_min = 5.0f; // used to check whether pt is off surface
    float thresh_offsurf_max = 100.0f; // used to check whether pt is off surface
    if (mode.compare("easy")==0) // the off surface will be very far away from surface
    {
        thresh_offsurf_min = 300.0f; // used to check whether pt is off surface
        thresh_offsurf_max = 1000.0f; // used to check whether pt is off surface
    }


    int *viewIdx;
    int viewIdx_visualization[] = {15};
    int viewNO = 0;

    int viewIdx_generation[64];
    for (int i=0;i<64;i++)
        viewIdx_generation[i] = i+1;

    if (visualization_ON)
    {
        viewIdx = viewIdx_visualization;
        viewNO = sizeof(viewIdx_visualization)/sizeof(viewIdx_visualization[0]);
    }
    else
    {
        viewIdx = viewIdx_generation;
        viewNO = sizeof(viewIdx_generation)/sizeof(viewIdx_generation[0]);
    }
    //int viewIdx[] = {15,16,17,23,24,25,26,32,33,34,35,36};

    std::stringstream pc_fileName;
    pc_fileName<<"/hdd1t/dataset/MVS/Points/stl/stl"<<std::setw(3)<<std::setfill('0')<<pc_Indx<<"_total.ply";

    std::stringstream output_fileName;
    if (visualization_ON)
    {
        std::cout << "now it's in the visualization mode!!!";
        output_fileName<<"/hdd1t/dataset/miniDataset/samplesTXT_visual/output_stl_"<<std::setw(3)<<std::setfill('0')<<pc_Indx<<"_"<<mode<<".txt";
    }
    else
        output_fileName<<"/hdd1t/dataset/miniDataset/samplesTXT/output_stl_"<<std::setw(3)<<std::setfill('0')<<pc_Indx<<"_"<<mode<<".txt";
    const std::string output_fileName_tmp = output_fileName.str();

    std::ofstream output_file;
    output_file.open(output_fileName_tmp.c_str(), ios::trunc);
    srand (static_cast <unsigned> (time(0)));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

    if (pcl::io::loadPLYFile (pc_fileName.str(), *source_cloud) < 0)  {
        std::cout << "Error loading point cloud ";
        return -1;
    }

    printf("\n width %d;\t height %d\n", source_cloud->width, source_cloud->height); //test

    //******* octree
    // Octree resolution - side length of octree voxels
    float resolution = 1.0f;
    //    int depth_2nd_last = 1; // closer to surface
    //    int depth_6th_last = 5; // little bit further to surface
    //    int depth_nth_last = octree.getTreeDepth()-1; // may be the farest
    int depth_indx = 7;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (resolution);
    octree.setInputCloud (source_cloud);
    octree.addPointsFromInputCloud ();

    int depth = octree.getTreeDepth()-depth_indx;

    Eigen::Matrix4d transform_M ;//= Eigen::Matrix4d::Identity();
    pcl::PointXYZRGB  cameraT;


    //******* generate on surface points

    // Check whether a voxel is visible in a view
    // if visible: red; not visible: green
    // Neighbors within voxel search
    pcl::PointXYZRGB Pt_onSurf;

    int PtIdx_onSurf =  1;
    int pt_imgx, pt_imgy; // the img postiion of the pt
    double ptdepth; // used to save the depth of current point

    for(int c=0; c<NO_onSurfPt;)
    {
        // the Pt_onSurf is a random point on surface.
        PtIdx_onSurf = randLong(1,source_cloud->width);
        Pt_onSurf = source_cloud->points[PtIdx_onSurf];

        // record on surface point: '1:viewIdx x y d visib;viewIdx x y d visib;viewIdx x y d visib'
        std::stringstream oneLineStr("");
        //        std::cout << "onelinstr:" << oneLineStr.str() << std::endl;
        oneLineStr << "1:";
        for(int indx=0; indx<viewNO; indx++)
        {
            //std::cout << "view index = " << viewIdx[indx] << "\t NO of views " << viewNO << "\n";
            read_cameraPO(viewIdx[indx], transform_M);
            read_cameraT(viewIdx[indx],cameraT);
            bool in_imgScope = Point_isinImgScope(Pt_onSurf,transform_M, pt_imgx, pt_imgy, ptdepth, img_x, img_y);

            if (in_imgScope)
            {
                bool voxel_visible = Voxel_isVisible(octree,Pt_onSurf,cameraT, thresh_occlusion);
                oneLineStr<< viewIdx[indx]<<" "<<pt_imgx<<" "<<pt_imgy<<" "<< ptdepth << " "<<int(voxel_visible)<<";";
                //std::cout << "\t current: " << oneLineStr.str() << "\n";
                if ( voxel_visible)
                {
                    //printf("%d ",c);
                    if (visualization_ON)
                        colorize_pt(source_cloud->points[PtIdx_onSurf], 0, 255, 0); // change rgb before add to viewer
                }
                else{
                    if (visualization_ON)
                        colorize_pt(source_cloud->points[PtIdx_onSurf], 255, 0, 0);
                    if (mode.compare("easy")==0) // in the 'easy' mode, the point should be visible in all views
                        break;
                    //std::cout<<"\n wrong PtIdx "<<PtIdx_onSurf<<std::endl;
                }
            }
            else break; // the point should in the scope of all the views

            if (indx==viewNO-1)
            {
                c++;
                printf(".");
                //std::cout << "onelinstr:" << oneLineStr.str() << std::endl;
                oneLineStr << Pt_onSurf.x <<" "<<Pt_onSurf.y <<" "<<Pt_onSurf.z;
                output_file << oneLineStr.str() << std::endl;
            }
        }
    }

    ///////////////////// the perspective projection /////////////////////
    // Executing the transformation
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
//    if (visualization_ON)
//    {
//        read_cameraPO(viewIdx[0], transform_M);
//        std::cout<<"testing"<<std::endl;
//        Cloud_Perspective_Projection(source_cloud, transformed_cloud, transform_M);
//    }
    ///////////////////// define visualizer ////////////////
    pcl::visualization::PCLVisualizer *viewer;
    if (visualization_ON)
    {
        viewer = new pcl::visualization::PCLVisualizer ("Matrix transformation example");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> source_cloud_color_handler(source_cloud);//, 255, 255, 255);
        // add the point clouds to the viewer and pass the color handler
        viewer->addPointCloud<pcl::PointXYZRGB> (source_cloud, source_cloud_color_handler, "original_cloud");

        //////////////////////// perspective projection /////////////////////
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
        read_cameraPO(viewIdx[0], transform_M);
        std::cout<<"testing"<<std::endl;
        Cloud_Perspective_Projection(source_cloud, transformed_cloud, transform_M);
//        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> transformed_cloud_color_handler(transformed_cloud);
//        viewer.addPointCloud<pcl::PointXYZRGB> (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

        viewer->addCoordinateSystem (100.0, "cloud", 0);
        viewer->setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    }


    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Iterator node_itr;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Iterator node_itr_end = octree.end();
    pcl::PointXYZRGB Pt_offSurf;

    std::cout << "\n===== Extracting data at depth " << depth << "... " << std::flush;
    unsigned long iterator_counter = 1; //used to count how many branches will be traversed.
    for (node_itr = octree.begin(depth); node_itr!= node_itr_end; ++node_itr)
        iterator_counter++;

    for(int j=0; j<NO_offSurfPt;)
    {
        GeneratePt_offSurface(depth,iterator_counter,octree,Pt_offSurf, thresh_offsurf_min, thresh_offsurf_max);
        // record on surface point: '1:viewIdx x y d visib;viewIdx x y d visib;viewIdx x y d visib'
        std::stringstream oneLineStr("");
        oneLineStr << "0:";

        for(int indx=0; indx<viewNO; indx++)
        {
            read_cameraPO(viewIdx[indx], transform_M);
            read_cameraT(viewIdx[indx],cameraT);
            bool in_imgScope = Point_isinImgScope(Pt_offSurf,transform_M, pt_imgx, pt_imgy, ptdepth, img_x, img_y);

            if (in_imgScope)
            {
                // check visiblity
                bool voxel_visible = Voxel_isVisible(octree,Pt_offSurf,cameraT, thresh_occlusion);
                oneLineStr<< viewIdx[indx]<<" "<<pt_imgx<<" "<<pt_imgy<<" "<<ptdepth<<" "<<int(voxel_visible)<<";";
                if(voxel_visible)
                {
                    if (visualization_ON)
                        viewer->addSphere (Pt_offSurf, 2, 0.0, 0.0, 0.8, randID("sphere_offsurf"));
                }
                else
                {
                    if (visualization_ON)
                        viewer->addSphere (Pt_offSurf, 2, 0.8, 0.0, 0.8, randID("sphere_offsurf"));
                }
            }
            else break; // the point should in the scope of all the views

            if (indx==viewNO-1)
            {
                j++;
                printf("@");
                oneLineStr << Pt_offSurf.x <<" "<<Pt_offSurf.y <<" "<<Pt_offSurf.z;
                //std::cout << "onelinstr:" << oneLineStr.str() << std::endl;
                output_file << oneLineStr.str() << std::endl;
            }
        }
    }

    if (visualization_ON)
    {
        while (!viewer->wasStopped ()) { // Display the visualiser until 'q' key is pressed
            viewer->spinOnce ();
        }
    }

    output_file.close();
    return 0;
}
