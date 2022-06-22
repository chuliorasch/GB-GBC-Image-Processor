from PIL import Image, ImageMath
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from mpl_toolkits import mplot3d
import math as mt

import itertools as it

start = time.time()


#-----------------------------------------------------------------------
#
#	27/04/2020
#
#	needs make data file for combinations, ram is not enough
#
#	progress: clustering done, need compare posible combinations
#
#
#	28/04/2020
#
#	need delete combination where repetion of colors < totalcolor 
#	
#
#	01/05/2020
#
#	extract first tile palettes and from that make comb
#
#
#	27/05/2020
#
#	extract all palettes and reduce them
#-----------------------------------------------------------------------

def KMeans_clust(X,out,flag):
	#Skmeans=KMeans(n_clusters=n_colors,precompute_distances=True,max_iter=500,algorithm='elkan',n_init=50).fit(X)
	kmeans=KMeans(n_clusters=n_colors).fit(X)

	#labs=kmeans.labels_
	if (flag == 1):
		print(len(X),' = Colors(points) originals')
		print(len(X)-len(color_length),' = Colors(points) converged')
		print(len(color_length),' = Final Colors(points) ')
		print('Colors out = ',n_colors,' / Max Colors Selected = ',max_colors)
		print('\n Starting KMs \n')
		
	for i in range(0,len(X)):
		out[i]=kmeans.cluster_centers_[kmeans.labels_[i]]
		
	return kmeans.cluster_centers_	
	
def clustering2(X,col,col_out):
	a=np.zeros((64,3))
	xyc2list(X,a)
	kmeans=KMeans(n_clusters=col).fit(a)

	for i in range(0,len(a)):
		a[i]=kmeans.cluster_centers_[kmeans.labels_[i]]
		
	list2xyc(a,X)
	col_out=kmeans.cluster_centers_
	return X

def GM_clust(X,out,flag): #X=list of colors,3 channels
	
	if (flag == 1):
		print(len(X),' = Colors(points) originals')
		print(len(X)-len(color_length),' = Colors(points) converged')
		print(len(color_length),' = Final Colors(points) ')
		print('Colors out = ',n_colors,' / Max Colors Selected = ',max_colors)
		print('\n Starting GMC \n')

	GM=GaussianMixture(n_components=n_colors,n_init=50,covariance_type='tied',tol=0.0001,max_iter=5000).fit(X) #tol=0.05
	gm_labs=GM.predict(X)
	means=GM.means_
	
	#print(GM.get_params())
	
	#---------------for GM---------------------------------
	for p in range(0,n_colors):
		for q in range(0,len(X)):
			for t in range(0,3):			
				if (gm_labs[q]==p):
					out[q,t]=int(means[p,t])
	
	return means
		
def plot(XX,centers,flag): #flag = true grapho full, flag = false graph centers
	
	fig=plt.figure()
	ax=plt.axes(projection='3d')
	
	if (flag==False):
		colors=centers.astype(int)/255
		for p in range(0,len(centers)):
			ax.scatter(centers[p,0],centers[p,1],centers[p,2],facecolor=colors[p,:])
	if (flag==True):
		colors=XX.astype(int)/255
		for p in range(0,len(XX)):
			ax.scatter(XX[p,0],XX[p,1],XX[p,2],facecolor=colors[p,:])
		ax.plot3D(centers[:,0],centers[:,1],centers[:,2],'xr') 

def color_pro(in_im,out_im):
	
	print('\n Starting Color Processing \n')
	
	tile_array=np.zeros((8,8,3), dtype=int)
	tile_array2=np.zeros((64,3), dtype=int)
	xtiles=int(in_im.shape[1]/8)
	ytiles=int(in_im.shape[0]/8)
	col_pal=np.zeros((4,3))
	pal=np.array([0,0,0,-1]).reshape(1,4)
	a=np.empty((0,4))
	b=0
	for m in range(0,ytiles):							#per y tile
		for l in range(0,xtiles):						#per x tile
			 		#rgb channel
			for j in range(0,8): 				#height
				for i in range(0,8): 			#width
						
					tile_array[j,i]=in_im[j+(m*8),i+(l*8)]
			
			xyc2list(tile_array,tile_array2)
			col_len=np.unique(tile_array2,axis=0)
			#print(col_len,"\n")				
			
			if (len(col_len) > 4):
				col_len2=np.zeros((4,3))
				max_col=4
				tile_array=clustering2(tile_array,max_col,col_len2)
			
			else:
				col_len2=np.full((len(col_len),3),non_color_3)
				
				for i in range(0,len(col_len)):
					col_len2[i]=col_len[i]
			
					
			
			col_len3=np.asarray(sorted(col_len2, key=lambda x: (x[0], x[1], x[2])))
			
			
			#---------------------------------------------------------------------------------------------------
			
			
			for r in range(0,len(np.unique(pal[:,3]))):
				for p in range(0,len(pal)):
					if (r==pal[p,3]):
						
						a=np.append(a,pal[p])
						
				
				if(np.isin(col_len3,a).all()==False):			
					pal=np.append(pal,np.append(col_len3,(r+1)*np.full((len(col_len3),1),1),axis=1),axis=0)
				
					if pal[0,3]==-1 and b==0:
						pal=np.delete(pal,0,0)
						b=1
					pal[:,3]=pal[:,3]+1		
						
			
			
			for j in range(0,8): 				#height
				for i in range(0,8): 
					out_im[j+(m*8),i+(l*8)]=tile_array[j,i]
	return pal			

def pal_pro(pal_l):
	final_pal=np.zeros((4,3))
	for i in range(0,int((len(pal_l))/4)):
		for j in range(0,4):
			temp_subpal=pal_l[j+i*4]
		
		#if (np.isin(final_pal,temp_subpal).all()==False):
		#	final_pal=np.append(final_pal,temp_subpal)
		#print(final_pal,'\n')
	return final_pal	
		
def del_alpha(im_alpha):
	for j in range(0,im_alpha.shape[1]):
		for i in range(0,im_alpha.shape[0]):
			if (im_alpha[i,j,3]<200):
				im_alpha[i,j]=non_color
								
def xyc2list(xyc,list_):
	for j in range(0,xyc.shape[0]):
		for i in range (0,xyc.shape[1]):
			list_[i+j*xyc.shape[1]]=xyc[j,i]
	
def list2xyc(list_,xyc):			
	for j in range(0,xyc.shape[0]):
		for i in range(0,xyc.shape[1]):
			xyc[j,i]=list_[i+j*xyc.shape[1]]	
					

#-----------------------------------------------------------------------

parser = argparse.ArgumentParser()#ArgumentParser(description='Short sample app')

parser.add_argument('in_name',type=str, help='Input png file')
parser.add_argument('--dithering', action="store_true",default=False,help='activate dithering')
parser.add_argument('--full', action="store_true",default=False,help='Display full Graphic')
parser.add_argument('--type', action="store",nargs='?',default='bkg',type=str,help='Define if is background [bkg] or Sprite [sp]')
parser.add_argument('--colors',action="store",nargs='?',type=int,default=32,help='Restrict max Colors')
parser.add_argument('--mode',action="store",nargs='?',type=int,default=1,help='Type of Clustering: Gaussian = 1, KMean = 0 (Default) ')
#parser.add_argument('colors_max', type=int,help='Add max Colors')

args = parser.parse_args()    
name_in=args.in_name
max_colors=args.colors
if (args.type=='sp'):
	if (max_colors>24):
		max_colors=24

non_color=[255,0,255,255]
non_color_3=[255,0,255]

im = Image.open("%s" %name_in).convert('RGBA')#.convert(colors=max_colors,dither=None,mode='P')#(rgba)
im_temp2 = np.array(im)


del_alpha(im_temp2)

im_temp=np.delete(im_temp2,3,axis=2)
im_shape=im_temp.shape #[ Height , Width , colors channels (RGBA)

#NOTE!: if max color=5 gets error cuz dont have many color to make 8 differents palettes

print('\n Image size ( Y , X , Channels ): ',im_shape,'\n')

if len(im_shape)<3:
	print("\n")
	print("    ---------------------------------------------")
	print("    Please be sure the input image is a PNG image")
	print("       the script can't find RGB/Alpha channels")
	print("    ---------------------------------------------")
	print("\n")
	quit()


#------------------------------------------------------
RRGGBB=np.zeros((im_shape[0]*im_shape[1],3))
RRGGBB2=np.zeros((im_shape[0]*im_shape[1],3))
new_nprgb=np.zeros(RRGGBB.shape, dtype=int)

new_rgb=np.zeros(im_shape, dtype=int)
new_rgb2=np.zeros(im_shape, dtype=int)
new_rgb3=np.zeros(im_shape, dtype=int)

palette=np.zeros((int(new_rgb.shape[0]/8),int(new_rgb.shape[1]/8),4,3))

pal_list=np.zeros((int(new_rgb.shape[0]/8)*int(new_rgb.shape[1]/8),4,3))
final_palette=np.zeros((4,3))

list_p=np.zeros((palette.shape[0]*palette.shape[1],4,3))
list_pp=np.zeros((list_p.shape[0]*4,3))

#max_colors_bg=int(8*4)#32 colors
#max_colors_sp=int(3*8)#24 colors

xyc2list(im_temp,RRGGBB)

color_length=np.unique(RRGGBB,axis=0)

#-------------------------------------------------------

if (len(color_length)< max_colors):
	n_colors=len(color_length)
	
else:	
	n_colors=max_colors	
		


#exit()

#------------------------------------------------------------------
if(args.mode==3):#NO
	cluster_centers=KMeans_clust(RRGGBB,RRGGBB2,1)
	cluster_centers=GM_clust(RRGGBB2,RRGGBB,1)
	RRGGBB2=RRGGBB
	
if(args.mode==2): #OK
	cluster_centers=GM_clust(RRGGBB,RRGGBB2,1)
	cluster_centers=KMeans_clust(RRGGBB2,RRGGBB,1)
	RRGGBB2=RRGGBB

if(args.mode==1):
	cluster_centers=GM_clust(RRGGBB,RRGGBB2,1)

if(args.mode==0):
	cluster_centers=KMeans_clust(RRGGBB,RRGGBB2,1)#ret cluster_center


list2xyc(RRGGBB2,new_rgb)	


#plot(RRGGBB,cluster_centers,args.full)
#plt.savefig('foo.png')
#plt.savefig('foo.pdf')
#plt.show()

#----------------------------------------------------------------

pale=color_pro(new_rgb,new_rgb3) #admit xyc

list2xyc(RRGGBB2,new_rgb2)

xyc2list(palette,pal_list) #cada 4 una paleta

print ('\n',pale)
print('\n',pale.shape,'\t',int(pale.shape[0]/4),'\n')
print('\t',len(np.unique(pale[:,3])),'\n')

new_im2=Image.fromarray(new_rgb3.astype('uint8'))#.quantize(colors=max_colors,method=2)
new_im2.save("clust0.png")

#--------------------------------------------------------------OK



#------------------------------------uncomment----------------------------------------------------

#new_im=Image.fromarray(new_rgb.astype('uint8'))#new_im = Image.fromarray(new_rgb.astype('uint8'))
#if (args.dithering==True):
#	new_im=new_im.convert(mode='P',colors=n_colors,dither=1)
#new_im.save("clust.png")
#new_im.show()

#---------------------------------------------------------------------------------------------
end = time.time()
print(end - start) 
