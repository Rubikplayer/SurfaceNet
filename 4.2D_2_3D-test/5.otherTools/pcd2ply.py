import thread
import os
import sys

"""
This piece of code tries to convert all the .pcd to .ply, and remove .pcd to save space
If the .ply file of the .pcd already exist, just remove the .pcd file

TODO: currently only do the job in one thread, latter can use 

usage
--------
python pcd2ply path
"""

def convt_pcd2ply(filePath_pcd):
    pcl_pcd2ply_exe = '~/Downloads/pcl-trunk/build/bin/pcl_pcd2ply'
    print("pcd2ply ing")
    if os.path.exists(filePath_pcd):
        filePath_ply = filePath_pcd.replace('.pcd','.ply')
        os.system( "{} {} {}".format(pcl_pcd2ply_exe, filePath_pcd, filePath_ply))
        if os.path.exists(filePath_ply):
            os.system("rm {}".format(filePath_pcd))
            print('file is removed. {}'.format(filePath_pcd))
    else:
        print('pcd2ply: file doesn\'t exist. {}'.format(filePath_pcd))
        return None
 

for root, dirs, files in os.walk(sys.argv[1]):
    for _file in files:
        if _file.endswith('pcd'):
            filePath_pcd = os.path.join(root, _file)
            file_ply = _file.replace('.pcd','.ply')
            filePath_ply = os.path.join(root, file_ply)
            if file_ply in files:
                # the corresponding ply file exist, only need to remove it
                os.system("rm {}".format(filePath_pcd))
                print('.ply already exist, remove pcd file: {}'.format(filePath_pcd))
                continue
            else:
                # the corresponding ply file DONOT exist, convert pcd2ply, then remove the pcd file
                # try:
                #     thread.start_new_thread(convt_pcd2ply, (filePath_pcd,))
                #     print 'converting pcd 2 ply in new thread'
                # except:
                #     print("Cannot start new thread to excecute pcd2ply. Run it in the orignal thread")
                convt_pcd2ply(filePath_pcd)






