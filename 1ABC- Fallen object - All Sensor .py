#!/usr/bin/env python
# coding: utf-8

# In[14]:


import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import scipy.spatial


# In[17]:


#Changeable Parameters

#Path where data is located
###############################################
dir_name= "C:\\Users\\918600\\OneDrive - Cognizant\\Desktop\\Project\\long_recording_BLK\\"

sensor_id=3  # 1- Velodyne,2- Faro ,3- Blickfeld, 4- Livox

#Rotation matrix parameter
rot_x1,rot_y1,rot_z1=1.4,2,0
rot_x2,rot_y2,rot_z2=0,0.05,0

#Cropping area Parameters
x1,y1,z1=0.209420,-10**5,-0.1498674 # set -10**5 if no value
x2,y2,z2=0.683784,2.9847,0.64047 #set 10**5 if no value

#RANSAC Algo. Parameters
distance_threshold=0.035

#DBSCAN Algo. Parameters
eps=0.045
min_points=5

#Measurement Parameters
min_labels=30
origin_x,origin_z=0.1221981,1.306213
sensor_height=0.87


######################################################

t_start=datetime.now()
loop=0
for filename in os.listdir(dir_name):
    filepath=os.path.join(dir_name,filename)
    #Reading pointcloud
    pcd=o3d.io.read_point_cloud(filepath)
    
    #Applying Rotation
    R=pcd.get_rotation_matrix_from_xyz((rot_x1 * np.pi,rot_y1 * np.pi,rot_z1 * np.pi))
    pcd.rotate(R)
    R=pcd.get_rotation_matrix_from_xyz((rot_x2 * np.pi,rot_y2 * np.pi,rot_z2 * np.pi))
    pcd.rotate(R)
    #o3d.visualization.draw_geometries([pcd],left=1000,top=200,width=800, height=800)
    df=pd.DataFrame(np.asarray(pcd.points))
    df.columns=['x','y','z']
    
    #Cropping Region of interest
    df=df[df['x']>x1]
    df=df[df['x']<x2]
    df=df[df['y']>y1]
    df=df[df['y']<y2]
    df=df[df['z']>z1]
    df=df[df['z']<z2]
    
    #Preparing data 
    pcd=o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(df.values)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    
    #Applying RANSAC 
    plane_model, inliers = pcd.segment_plane(distance_threshold,ransac_n=3,num_iterations=10)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    #Applying DBSCAN

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            outlier_cloud.cluster_dbscan(eps, min_points))
    
    #Isolating Objects
    if labels.shape[0]>0:
        max_label = labels.max()
        df=pd.DataFrame(labels)
        label_count=df.value_counts()
        total_object=[]
        labl=range(max_label+1)
        for i in range(max_label+1):
            if label_count[i]>min_labels:
                total_object.append(i)
        point_clouds=[]
        for j in total_object:
            pcd=o3d.geometry.PointCloud()

            pcd.points=o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points)[labels==j])
            point_clouds.append(pcd)
        print("")
        ori =pd.DataFrame(columns=['time','object','height','width','depth','volume','location','surface_area','tot_obj'])
        if len(point_clouds)>=1:
            print("Fallen Object Detected")
            
            print(" ")
            count=1
            for xx in point_clouds:
                # Length and Width
                obj=np.array(xx.points)
                height_min=obj[:,1].min()
                height_max=obj[:,1].max()
                width_min=obj[:,0].min()
                width_max=obj[:,0].max()
                depth_min= obj[:,2].min()
                depth_max= obj[:,2].max()
                depth=abs(depth_max-depth_min)
                height,width=abs(height_max-height_min),abs(width_max-width_min)

                if height<width:
                    height,width=width,height

                print("Height: ",height*100,"cm")
                print("Width: ",width*100,"cm")
                print("Depth: ",depth*100,"cm")
                #Centroid of object in X-Z plane
                def centroid(arr):
                    length = arr.shape[0]
                    ln=length//2
                    return np.sort(arr[:,0])[ln], np.sort(arr[:,2])[ln]

                cent=centroid(obj)

                #Sesnor to object Distance
                '''
                xdis=np.array(inlier_cloud.points)
                half_x=abs(xdis[:,0].min()-xdis[:,0].max())/2
                mid_edge_2_perp=abs(half_x-cent[0])
                if sensor_id==1:
                    min_z=xdis[:,2].min()
                else:
                    min_z=xdis[:,2].max()

                zdis=abs(min_z-cent[1])
                mid_edge_2_obj=np.square(zdis)+np.square(mid_edge_2_perp)
                sensor_2_obj=np.sqrt(mid_edge_2_obj+np.square(sensor_height))
                print("Object to Sensor Distance",sensor_2_obj*100)
                '''
                mid_edge_2_obj=np.square(cent[0]-(origin_x-0.5))+np.square(cent[1]-origin_z)
                sensor_2_obj=np.sqrt(mid_edge_2_obj+np.square(1.52))
                print("Object to Sensor Distance",sensor_2_obj*100)
                #################################################
                obj=np.array(inlier_cloud.points)
                if sensor_id==1:
                    origin_x=obj[:,0].min()
                    origin_z=obj[:,2].min()

                #print(cent)
                if sensor_id==1:
                    loc=((cent[0]-origin_x)*100,(cent[1]-origin_z)*100)
                else:
                    loc=(-(cent[1]-origin_z)*100,-(cent[0]-origin_x)*100)

                print("Location of Object: ",loc)
                print(" ")
                print(" ")
                surf_area=height*width*10000
                arr={'time':str(datetime.now()),'object':"Object"+str(count),'height':height*100,'width':width*100,'depth':depth*100,'volume':height*width*depth*1000000,'location':loc,'sensor_2_obj':sensor_2_obj*100,'surface_area':surf_area,'tot_obj':len(point_clouds)}
                ori=ori.append(arr,ignore_index=True)

                ##print(json_df)
                ori.to_json('json_data/fallen'+str(loop)+'.json',orient='records')
                
        else:
            arr={'time':str(datetime.now()),'object':"No Object",'height':0,'width':0,'depth':0,'volume':0,'location':0,'sensor_2_obj':0,'surface_area':0,'tot_obj':0}
            ori=ori.append(arr,ignore_index=True)

            ##print(json_df)
            ori.to_json('json_data/fallen'+str(loop)+'.json',orient='records')
            
    #print(filepath)
    loop+=1
t_end=datetime.now()
e=t_end-t_start
print('Execution Time',e)


# In[18]:


o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud],left=1000,top=200,width=800, height=800)





