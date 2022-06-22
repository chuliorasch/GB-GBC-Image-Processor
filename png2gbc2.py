#tomar colores mas significativos de cada tile 
#comparar con otros set de tiles
#tomarlos valores mas significativos y armar las paletas en base a eso
#
#
#op2:
#tomar mean shift segmentation y armar a partir de ahi
#
#


# Using PILLOW

from PIL import Image
import numpy as np
import argparse
import time

start = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('in_name',type=str, help='Input png file')
parser.add_argument('out_name', type=str,help='Output png file')

args = parser.parse_args()    
name_in=args.in_name
name_out=args.out_name



im = Image.open("%s" %name_in).convert('RGBA')
imm= Image.open("%s"%name_in).convert('P')
imm.save("totosme.png",bit="1")
#----------------------------------------------------------------------

#imageWithColorPalette = im.convert("P", palette=Image.ADAPTIVE, colors=52)

#imageWithColorPalette.show()

#np_im = np.array(imageWithColorPalette)

#print(np_im.shape)
#----------------------------------------------------------------------

np_im= np.array(im)
im_shape=np_im.shape #[ Height , Width , colors channels (RGBA)

if len(im_shape)<3:
	print("\n")
	print("    ---------------------------------------------")
	print("    Please be sure the input image is a PNG image")
	print("       the script can't find RGB/Alpha channels")
	print("    ---------------------------------------------")
	print("\n")
	quit()

minmax_val=np.zeros((3,2), dtype=int)
minmax_val_7pal=np.zeros((3,2), dtype=int)
minmax7gray=[0,0]
tile_array=np.zeros((8,8,3), dtype=int)
tile_array2=np.zeros((8,8,3), dtype=int)
im_031=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
im_031_4cpt=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
im_031_4cpt_7pal=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
im_array=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
im_gray7=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
im_gray4=np.zeros((im_shape[0],im_shape[1],3), dtype=int)
delta_RGB=np.zeros(3, dtype=int)
delta_RGB_7pal=np.zeros(3, dtype=int)


xtiles=int(im_shape[0]/8)
ytiles=int(im_shape[1]/8)



#----------------------------------------------------------------------

#imageWithColorPalette = im.convert("P", palette=Image.ADAPTIVE, colors=52)

#imageWithColorPalette.show()

#np_im = np.array(imageWithColorPalette)


#----------------------greyscaled 4 color------------------------------------------
def grayscale():           
	for i in range(0,im_shape[0]): #height
		for j in range(0,im_shape[1]): #width
			u=(float(np_im[i,j,0])+float(np_im[i,j,1])+float(np_im[i,j,2]))/3
			grey_val=1/3*np.heaviside(u-(256/6)*1,1)+1/3*np.heaviside(u-(256/6)*3,1)+1/3*np.heaviside(u-(256/6)*5,1)
			im_gray4[i,j,0]=int(255*grey_val)
			im_gray4[i,j,1]=int(255*grey_val)
			im_gray4[i,j,2]=int(255*grey_val)
			#print(int(255*grey_val))
	#grey_map_data
	
	
#grayscale()

#----------------------greyscaled 7 color------------------------------------------
def grayscale7(old_img):  
	
	aux=0

	minmax7gray[0]=np.amin(old_img,axis=(0,1,2))
	minmax7gray[1]=np.amax(old_img,axis=(0,1,2))
	
	delta7gray=minmax7gray[1]-minmax7gray[0]
	
	#print(minmax7gray, delta7gray)
		
	for j in range(0,im_shape[1]): 				#height
		for i in range(0,im_shape[0]):
					
			u=(old_img[i,j,0]+old_img[i,j,1]+old_img[i,j,2])/3
					
			for ppp in range(1,7):
				
				#if minmax7gray[0]==0:
				#	minmax7gray[0]=1
					
				ppp=(ppp*2-1)
				aux=aux+delta7gray*np.heaviside(u-((delta7gray/12)*ppp+minmax7gray[0]),1)/6
			
			im_gray7[i,j,0]=aux+minmax7gray[0]
			im_gray7[i,j,1]=aux+minmax7gray[0]
			im_gray7[i,j,2]=aux+minmax7gray[0]
			aux=0
			#print(int(255*grey_val))
	#grey_map_data
	
	


#---------------------------------------Direct 4 colors per tile ----------------------------------------------------------------
def direct_4(old_img):
	aux=0
	for m in range(0,ytiles):							#per y tile
		for l in range(0,xtiles):						#per x tile
			for k in range(0,3): 		#rgb channel
				for j in range(0,8): 				#height
					for i in range(0,8): 			#width
						
						tile_array[i,j,k]=old_img[i+(l*8),j+(m*8),k] 
						
						for ppp in range(1,32):
						
							ppp=(ppp*2-1)
							aux=aux+np.heaviside(u-(256/64)*ppp,1) # rango de 0 a 31 para rango de 0 a 1 en 1/32 de fraccion hay que multiplicar el heaviside por 1/32
							
						tile_array[i,j,k]=aux
						aux=0
						
						#------------------- 0 to 31 correction done--------------------
						#------------------- tile array go 0 to 31 ---------------------
						
				#color channel +1
			#all channels done
			
			minmax_val[:,0]=np.amin(tile_array, axis=(0,1))
			minmax_val[:,1]=np.amax(tile_array, axis=(0,1))
			
			delta_RGB[0]=minmax_val[0,1]-minmax_val[0,0] #delta
			delta_RGB[1]=minmax_val[1,1]-minmax_val[1,0]
			delta_RGB[2]=minmax_val[2,1]-minmax_val[2,0]
			
			for p in range(0,3): 		#rgb channel
				for q in range(0,8): 				#height
					for r in range(0,8):			#width
						
						u=tile_array[r,q,p]
						
						for ppp in range(1,4):
						
							ppp=(ppp*2-1)
							aux=aux+delta_RGB[p]*np.heaviside(u-((delta_RGB[p]/6)*ppp+minmax_val[p,0]),1)/3 # rango de 0 a 31 o rango de entrada
							#print(ppp)												

						im_031_4cpt[r+(l*8),q+(m*8),p]=aux+minmax_val[p,0]
						aux=0				


#----------------------------------------------------------------------------------------------------
def color_processing_2():							#Procesa 4 colores per tile
	aux=0
	for m in range(0,ytiles):							#per y tile
		for l in range(0,xtiles):						#per x tile
			for k in range(0,3): 		#rgb channel
				for j in range(0,8): 				#height
					for i in range(0,8): 			#width
						
						u=np_im[i+(l*8),j+(m*8),k] 
						
						for ppp in range(1,32):
						
							ppp=(ppp*2-1)
							aux=aux+np.heaviside(u-(256/64)*ppp,1) # rango de 0 a 31 para rango de 0 a 1 en 1/32 de fraccion hay que multiplicar el heaviside por 1/32
							
						tile_array[i,j,k]=aux
						tile_array2[i,j,k]=tile_array[i,j,k]
						#im_031[i+(l*8),j+(m*8),k]=aux #non edited 0 to 31
						#im_array[i+(l*8),j+(m*8),k]=aux*8
						aux=0
						
						#------------------- 0 to 31 correction done--------------------
						#------------------- tile array go 0 to 31 ---------------------
						
				#color channel +1
			#all channels done
			
			minmax_val[:,0]=np.amin(tile_array, axis=(0,1))
			minmax_val[:,1]=np.amax(tile_array, axis=(0,1))
			
			delta_RGB[0]=minmax_val[0,1]-minmax_val[0,0] #delta
			delta_RGB[1]=minmax_val[1,1]-minmax_val[1,0]
			delta_RGB[2]=minmax_val[2,1]-minmax_val[2,0]
			
			for p in range(0,3): 		#rgb channel
				for q in range(0,8): 				#height
					for r in range(0,8):			#width
						
						u=tile_array[r,q,p]
						
						for ppp in range(1,4):
						
							ppp=(ppp*2-1)
							aux=aux+delta_RGB[p]*np.heaviside(u-((delta_RGB[p]/6)*ppp+minmax_val[p,0]),1)/3 # rango de 0 a 31 o rango de entrada
							#print(ppp)												

						im_031_4cpt[r+(l*8),q+(m*8),p]=aux+minmax_val[p,0]
						aux=0				
			
	
	#--------------------------------- 4 color per tiles done -------------------------------------
	#---------------------------------- processing 7 palettes -------------------------------------
	
	
	minmax_val_7pal[:,0]=np.amin(im_031_4cpt, axis=(0,1))
	minmax_val_7pal[:,1]=np.amax(im_031_4cpt, axis=(0,1))
			
	delta_RGB_7pal[0]=minmax_val_7pal[0,1]-minmax_val_7pal[0,0] #delta
	delta_RGB_7pal[1]=minmax_val_7pal[1,1]-minmax_val_7pal[1,0]
	delta_RGB_7pal[2]=minmax_val_7pal[2,1]-minmax_val_7pal[2,0]
		
	for k in range(0,3): 		#rgb channel
		for j in range(0,im_shape[1]): 				#height
			for i in range(0,im_shape[0]):
						
				u=im_031_4cpt[i,j,k]
						
				for ppp in range(1,7):
						
					ppp=(ppp*2-1)
					aux=aux+delta_RGB_7pal[p]*np.heaviside(u-((delta_RGB_7pal[p]/12)*ppp+minmax_val_7pal[p,0]),1)/6
						
				im_031_4cpt_7pal[i,j,k]=aux+minmax_val_7pal[p,0]
				aux=0		
										
						
											
	#print(delta_RGB)
	#print(np.amin(tile_array2, axis=(0,1)),np.amax(tile_array2, axis=(0,1)))
	#print(np.amin(tile_array, axis=(0,1)),np.amax(tile_array, axis=(0,1)))
	#print(np.amin(im_031_4cpt, axis=(0,1)),np.amax(im_031_4cpt, axis=(0,1)))
	
	
	
#-------------------------------------------------direct 7 ----------------------------------------

def direct_7():							#Procesa 4 colores per tile
	for k in range(0,3):
		for j in range(0,im_shape[1]):
			for i in range(0,im_shape[0]):
				im_031[i,j,k]=np_im[i,j,k]
				
	aux=0
	minmax_val_7pal[:,0]=np.amin(im_031, axis=(0,1))
	minmax_val_7pal[:,1]=np.amax(im_031, axis=(0,1))
	
	delta_RGB_7pal[0]=minmax_val_7pal[0,1]-minmax_val_7pal[0,0] #delta
	delta_RGB_7pal[1]=minmax_val_7pal[1,1]-minmax_val_7pal[1,0]
	delta_RGB_7pal[2]=minmax_val_7pal[2,1]-minmax_val_7pal[2,0]
	
	for k in range(0,3): 		#rgb channel
		for j in range(0,im_shape[1]): 				#height
			for i in range(0,im_shape[0]): 		#width
								
					u=im_031[i,j,k]
					
					for ppp in range(1,7):
								
						ppp=(ppp*2-1)
						aux=aux+delta_RGB_7pal[k]*np.heaviside(u-((delta_RGB_7pal[k]/12)*ppp+minmax_val_7pal[k,0]),1)/6
								
					im_031_4cpt_7pal[i,j,k]=aux+minmax_val_7pal[k,0]
					aux=0
					##########################
#					u=np_im[i+(l*8),j+(m*8),k] 
						
#					for ppp in range(1,32):
#						
#						ppp=(ppp*2-1)
#						aux=aux+np.heaviside(u-(256/64)*ppp,1) # rango de 0 a 31 para rango de 0 a 1 en 1/32 de fraccion hay que multiplicar el heaviside por 1/32
#							
#					tile_array[i,j,k]=aux
#					tile_array2[i,j,k]=tile_array[i,j,k]
#					#im_031[i+(l*8),j+(m*8),k]=aux #non edited 0 to 31
#					#im_array[i+(l*8),j+(m*8),k]=aux*8
#					aux=0
#						
#						#------------------- 0 to 31 correction done--------------------
						#------------------- tile array go 0 to 31 ---------------------
						
				#color channel +1
			#all channels done
			
		
											
	
	#--------------------------------- 4 color per tiles done -------------------------------------
	#---------------------------------- processing 7 palettes -------------------------------------
	
	
				
						
											
	#print(delta_RGB)
	#print(np.amin(tile_array2, axis=(0,1)),np.amax(tile_array2, axis=(0,1)))
	#print(np.amin(tile_array, axis=(0,1)),np.amax(tile_array, axis=(0,1)))
	#print(np.amin(im_031_4cpt, axis=(0,1)),np.amax(im_031_4cpt, axis=(0,1)))
	

#------------------------------------------------ Std Weighted-------------------------------------------------------------------

#sigma=np.std(tile_array, axis=(0,1))
#sigma_x=sigma/8
#
#sigma_i=val()/64
#
#for i in range(0,64)
#
#	mean_wvalue=sigma_x*sigma_x*sum(val()/(sigma_i*sigma_i))
#


#---------------------------------------Exe-------------------------------------------------------			

#color_processing_2()  	#im_031 & im_array
direct_7()		
#grayscale()				#im_gray4

#grayscale7(im_031_4cpt_7pal)			#im_gray7	
#asm_output(im_gray4)	im -> asm		

#--------------------------------------heaviside-----------------------------------------------
#                      0   if x1 < 0
#heaviside(x1, x2) =  x2   if x1 == 0
#                      1   if x1 > 0
#examples:
#>>> np.heaviside([-1.5, 0, 2.0], 0.5)
#array([ 0. ,  0.5,  1. ])
#>>> np.heaviside([-1.5, 0, 2.0], 1)
#array([ 0.,  1.,  1.])                      

#------------------------------------ ^ don't delete ^ -------------------------------------------

#color correction has been done
#r g b 64 pixels
# avanza tile por tile, en el orden del gameboy, examina cada tile 
# y encuadra los colores por tile

def bg_color_processing():							#Procesa 4 colores per tile
	for m in range(0,32):							#per y tile
		for l in range(0,32):						#per x tile
			for k in range(0,3): 		#rgb channel
				for i in range(0,8): 				#height
					for j in range(0,8): 			#width
						np_im[i+(l*8),j+(m*8),k]
				
				#for n in range(0,4):# n = color index in palette [0 to 3]					
					#tile_map_rgb[l,m,n,k]=sum_val/64	





#repeated/flipped tile:

#normal scan i,j 		flag 00
#y flip scan 7-i,j		flag 10
#x flip scan i,7-j		flag 01
#xy flip scan 7-i,7-j	flag 11



#repeated palette
#condensed 7 palettes

#------------------------------------------------------------------------------------
im_031_4cpt=(im_031_4cpt+1)*8-1
im_031_4cpt_7pal=(im_031_4cpt_7pal+1)*8-1
im_gray7=(im_gray7+1)*8-1
#print(np.amin(im_031_4pt, axis=(0,1)),np.amax(im_031_4pt, axis=(0,1)))

end = time.time()
print(end - start)       


new_im = Image.fromarray(im_gray7.astype('uint8'))
new_im.save("gray7.png")
#new_im = Image.fromarray(im_gray4.astype('uint8'))
#new_im.save("gray4.png")
new_im = Image.fromarray(im_031_4cpt.astype('uint8'))
new_im.save("4cpt.png")
new_im = Image.fromarray(im_031_4cpt_7pal.astype('uint8'))
new_im.save("%s"%name_out)


#Image.fromarray(im_031_4cpt).show()
#Image.fromarray(im_031_4cpt_7pal).show()
#Image.fromarray(im_gray7).show()

#--------------------------------- Write to asm text -------------------------
def asm_output(image_data):
	
	output=open('output.inc','w')
	#im_shape
	for i in range(0,len(data)):
		
			new_data[i]=data[i].rstrip('\n\t').lstrip().replace("0x","$")
			
	
	for k in range(0,int(len(new_data)/8)):
		
		#output.write('DB\t$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s,$%s'%image_data[])
		
		for j in range(0,8):
			
			if j<7:
				
				output.write(new_data[j+k*8])
				output.write(',')
				
			else:
				output.write(new_data[j+k*8])
				output.write('\n')
			
	
