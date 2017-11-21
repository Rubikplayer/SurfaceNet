/* \author Geoffrey Biggs */


#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <fstream>

inline
float randFloat(float LO, float HI) {return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));}


boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewers (pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clouds[], int N)
{

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->initCameraParameters ();
  int vs[N];

  for(int i=0; i<N; i++)
  {
      std::ostringstream subviewer_name, txt_name, normal_name;
      subviewer_name << "subviewer" << i;
      txt_name << "txt" << i;
      normal_name << "normal" << i;
//      std::cout << subviewer_name.str() << "  " << txt_name.str() << std::endl;
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = clouds[i];
      viewer->createViewPort(i*1.0/N, 0.0, (i+1)*1.0/N, 1.0, vs[i]); //position of the ports: it's xmin, ymin, xmax, ymax
      float gray_value = 0.03*(i+1);
      viewer->setBackgroundColor (gray_value, gray_value, gray_value, vs[i]);
      viewer->addText("Radius: 0.1", 10, 10, txt_name.str(), vs[i]);
//      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBNormal> single_color(cloud, 0, 255, 0);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(cloud);
      viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, rgb, subviewer_name.str(), vs[i]);


      // thanks: http://www.pcl-users.org/Visualize-PointXYZINormal-td4031160.html //normal_name.str(),vs[i] IMPORTANT
      viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (cloud, cloud, 10, 0.5, normal_name.str(),vs[i]);

      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, subviewer_name.str());
  }
  viewer->addCoordinateSystem (1.0);
  return (viewer);
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr parse_txt_cloud (char* str)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr txt_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointXYZRGBNormal basic_point;
    std::ifstream infile(str);

    float resolution = 0.5f;
    float half_resol = resolution/2;
    int x,y,z,R,G,B,density;
    float normalx=0,normaly=0,normalz=0;

    float cx,cy,cz;

    std::string line;
    std::vector<float> vec;
    float value; // auxiliary variable to which you extract float from stringstream

    while (getline(infile, line))
    {
        std::istringstream iss(line);
        while(iss >> value)        // yields true if extraction succeeded
            vec.push_back(value);  // and pushes value into the vector
        x = vec[0];
        y = vec[1];
        z = vec[2];
        R = vec[3];
        G = vec[4];
        B = vec[5];
        density = vec[6];
        if(vec.size()>7)
        {
            normalx = vec[7];
            normaly = vec[8];
            normalz = vec[9];
        }

//        while (infile >> x >> y >> z >> R >> G >> B >> density >> normalx >> normaly >> normalz)
        vec.clear(); // IMPORTANT

        cx = x*resolution;
        cy = y*resolution;
        cz = z*resolution;
        basic_point.r = R; //density * 10;
        basic_point.g = G; //100 + density * 10;
        basic_point.b = B; //density * 10;
        basic_point.normal_x = normalx;
        basic_point.normal_y = normaly;
        basic_point.normal_z = normalz;
        for(int n=0; n<density; n++)
        {
            basic_point.x = randFloat(cx-half_resol,cx+half_resol);
            basic_point.y = randFloat(cy-half_resol,cy+half_resol);
            basic_point.z = randFloat(cz-half_resol,cz+half_resol);
            txt_cloud_ptr->points.push_back(basic_point);
        }

    }
    return txt_cloud_ptr;
}

// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
  int N=argc-1;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr point_clouds[N];

  for(int i=0; i<N; i++)
  {
    point_clouds[i] = parse_txt_cloud(argv[i+1]);
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = multiViewers(point_clouds, N);


  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}
