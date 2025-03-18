#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "timer.h"

//1. Define a set of initial particles
//2. Calculate the forces on each of those particles
//3. Update the particles with how the force on them affects their position and velocity
//4. Repeat with the new set of particles


//This is the kernel (where the calculation happens)
__global__ void force(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *m, float dt, int n, double G){	  
	   int t = blockIdx.x*blockDim.x +threadIdx.x;
	   int T = blockDim.x*gridDim.x;

	   //create loop to calculate forces
	   for(int i = t; i < n; i+=T){
	   	//initialise forces to 0
	      float Fx = 0.0;
	      float Fy = 0.0;
	      float Fz = 0.0;

	      //find forces for each of the n bodies
	      for(int j =0; j<n; j++){
	      		//skip forces between particle and itself
		      if(j==i){
		      		continue;}
	      	      //calculate distance between particle and neighbours for each of the particles
	      	      float dx = x[j] - x[i]; 
		      float dy = y[j] - y[i];
		      float dz = z[j] - z[i];

		      //find total distance by squaring
		      float totdist = rsqrt(dx*dx + dy*dy + dz*dz);
		      float distcube = totdist*totdist*totdist;

		      //find the forces and add them on to the initial value
		      Fx += G*m[j]*dx*distcube;
		      Fy += G*m[j]*dy*distcube;
		      Fz += G*m[j]*dz*distcube;
		      }
		      
	       //update the velocity values from the new forces
	       vx[i] += dt*Fx;
	       vy[i] += dt*Fy;
	       vz[i] += dt*Fz;

		//update position values from the new forces
	       x[i] += vx[i]*dt;
	       y[i] += vy[i]*dt;
	       z[i] += vz[i]*dt;
	       }

}

int main(int argc, char *argv[]){

    //make file for snapshots
    FILE *fptr1, *fptr2, *fptr3;

	//USER-INPUT DEFINITIONS
    //define number of bodies
    int nbodies;
    sscanf(argv[1], "%i", &nbodies);

    //define size of timestep
    float dt;
    sscanf(argv[2],"%f", &dt);
  
    //define number of iterations
    int iter;
    sscanf(argv[3],"%i", &iter);

    printf("Number of bodies: %i Number of iterations: %i\n", nbodies, iter);

    dt *= 3*pow(10, 7);

    //start timer
    float t_tot = 0.0f;

    //define memory values on host
    int bytes = nbodies*sizeof(float);
    float *x = (float*)malloc(bytes);
    float *y = (float*)malloc(bytes);
    float *z = (float*)malloc(bytes);
    float *vx = (float*)malloc(bytes);
    float *vy = (float*)malloc(bytes);
    float *vz = (float*)malloc(bytes);
    float *m = (float*)malloc(bytes);
   
    for (int i = 0; i < nbodies; ++i) {
    	x[i] = rand()/(float)RAND_MAX*2*10000 - 10000;
    	y[i] = rand()/(float)RAND_MAX*2*10000 - 10000;
    	z[i] = rand()/(float)RAND_MAX*2*10000 - 10000;
    	vx[i] = 0.f; //rand()/(float)RAND_MAX*2*1 - 1;
    	vy[i] = 0.f; //rand()/(float)RAND_MAX*2*1 - 1;
    	vz[i] = 0.f; //rand()/(float)RAND_MAX*2*1 - 1;
    	m[i] = rand()/(float)RAND_MAX*pow(10, 5);
  }
	
    fptr1 = fopen("/cosma8/data/dp004/dc-john7/snapshot_0.txt", "w");
    for(int i = 0; i<nbodies; i++){
            	fprintf(fptr1, "%f %f %f %f %f %f %f\n", x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);}		    
    fclose(fptr1);
    

    float G = 6.67*pow(10, -11);

    //allocate device memory size
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_vx, bytes);
    cudaMalloc(&d_vy, bytes);
    cudaMalloc(&d_vz, bytes);
    cudaMalloc(&d_m, bytes);

    //copy memory to device
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, bytes, cudaMemcpyHostToDevice);

    for(int j =1; j<= iter; j++){
	    if(j%10==0){
		printf("Running %i \n", j);}  

	    StartTimer();	   
	    
	    //do calculation
	    force<<<dim3(256,1,1),dim3(128,1,1)>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_m, dt, nbodies, G);

	    cudaError_t err = cudaGetLastError();
	    if (err != cudaSuccess) 
    	       printf("Error: %s\n", cudaGetErrorString(err));

	    cudaDeviceSynchronize();  

	    //copy memory back to host
	    cudaMemcpy(x, d_x, bytes, cudaMemcpyDeviceToHost);
	    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);
   	    cudaMemcpy(z, d_z, bytes, cudaMemcpyDeviceToHost);
	    cudaMemcpy(vx, d_vx, bytes, cudaMemcpyDeviceToHost);
	    cudaMemcpy(vy, d_vy, bytes, cudaMemcpyDeviceToHost);
	    cudaMemcpy(vz, d_vz, bytes, cudaMemcpyDeviceToHost);

	    /*for(int i = 0; i<nbodies; i++){
            printf("Particle number %i- ", i);
            printf("x:%f, y:%f, z:%f, vx:%f, vy:%f, vz:%f, ", x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);
            printf("\n");}*/

	    const float tElapsed = GetTimer();
	    t_tot += tElapsed;
	
	    char filename[100];
	    snprintf(filename, sizeof(char) * 100, "snapshot_%i.txt", j);
	
	    fptr2 = fopen(filename, "w");
	    for(int i = 0; i<nbodies; i++){
            	fprintf(fptr2, "%f %f %f %f %f %f %f\n", x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);}
	    fclose(fptr2);


	}

	printf("Total time: %.3f ms\n", t_tot);

        /*char filename[100];
  	snprintf(filename, sizeof(char) * 32, "benchmarking-2.csv");

  	char hostname[1024];
  	hostname[1023] = '\0';
  	gethostname(hostname, 1023);
  	strtok(hostname, ".");

  	fptr3 = fopen(filename, "a");
  	fprintf(fptr3, "%s %s %i %f %i %f\n", hostname, "cuda", nbodies, dt/(3*pow(10, 7)), iter, t_tot);
  	fclose(fptr3);*/

	return 0;
}

