/*

Naive C Implementation of a Mixed Logit w/ Individual Data 

Utility specification: 
 U_{ijt} = \beta_{0} + \beta_{1}*X_{j} + \beta_{2}*W_{i}*X_{j} +\beta_{3}*S_{i}*X_{j} + \eps_{i,j}

Notes: 
 X is a single product characterstic 
 W is a single consumer specific variable 
 S is a single unobserved taste shock, constant across products and individuals 

After compiling the program (I used standard gcc), the program takes in 3 arguments;
  N_CONS - Number of consumers (100000)
  N_CHOICES - Number of products (10)
  N_SIMS - Number of simulation draws (1000)

e.g, 

mixedLogit 100000 10 1000

*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv){

  int N_CONS = atoi(argv[1]);
  int N_CHOICES = atoi(argv[2]);
  int N_SIMS = atoi(argv[3]);
  printf("\n N_CONS = %d \n N_CHOICES = %d \n N_SIMS = %d \n", N_CONS, N_CHOICES, N_SIMS); 

  /*  FIRST DRAW DGP */

  // Initalize Data Objects  
  double *X = malloc(N_CHOICES*sizeof(double));
  double *W = malloc(N_CONS*sizeof(double));
  double *S = malloc(N_SIMS*sizeof(double));
  int *Y = malloc(N_CONS*sizeof(int));

  // Set Seed 
  srand((unsigned int) time(NULL));

  // Draw U ~ [0,1] for product chars 
  for (int i = 0; i < N_CHOICES; i++){
   X[i] = (float) rand() / (float)(RAND_MAX);
   //printf("%f\n", X[i]);
  }

  // Draw U ~ [0,1] for individual chars 
  for (int i = 0; i < N_CONS; i++){
   W[i] = (float) rand() / (float)(RAND_MAX);
   //printf("%f\n", W[i]);
  }  

  // Draw U ~ [0,1] for random utility draws 
  for (int i = 0; i < N_SIMS; i++){
   S[i] = (float) rand() / (float)(RAND_MAX);
   //printf("%f\n", S[i]);
  }

  // Draw a consumer's choice as a random integer 
  for (int i = 0; i < N_CONS; i++){
   Y[i] = (float) (rand() % N_CHOICES + 1);
   // printf("%f\n", Y[i]);
  }

  // Draw preference parameters from a uniform distribution as well 
  float beta0 = (float) rand() / (float)(RAND_MAX);
  float beta1 = (float) rand() / (float)(RAND_MAX);
  float beta2 = (float) rand() / (float)(RAND_MAX);
  float beta3 = (float) rand() / (float)(RAND_MAX);

  /*  FINDING THE LOG LL */
  clock_t tic = clock();

  // Calculate exponetial of the utils 
  double (*ExpUtils)[N_CHOICES][N_SIMS] = malloc(sizeof(double[N_CHOICES][N_SIMS][N_CONS])); 
  // double ExpUtils[N_CHOICES][N_SIMS][N_CONS];
  for (int i = 0; i < N_CHOICES; i++){
    for (int j = 0; j < N_SIMS; j++){
      for (int k = 0; k < N_CONS; k++){
        ExpUtils[i][j][k] = exp(beta0 + beta1*X[i] + beta2*X[i]*W[k] + beta3*S[j]*X[i]); 
      }
    }
  }
 
  // Caclulate shares given each draw of the sim var
  // double shares[N_CHOICES][N_CONS][N_SIMS];
  double (*shares)[N_CHOICES][N_SIMS] = malloc(sizeof(double[N_CHOICES][N_SIMS][N_CONS]));
  for (int j = 0; j < N_SIMS; j++){
    for (int k = 0; k < N_CONS; k++){
        // calculate the denominator 
        double d = 0; 
        for (int i = 0; i < N_CHOICES; i++){
            d += ExpUtils[i][j][k];
        } 
        // calculate the shares over each simulation draw, given the respective denom 
        for (int i = 0; i < N_CHOICES; i++){
            shares[i][j][k] = ExpUtils[i][j][k]/d;
            // printf("%f\n", ExpUtils[i][0][0]/d);
        } 
    }
  }

  // Integrate over the shares to get CCPs of individuals
  double (*CCP)[N_CHOICES] = malloc(sizeof(double[N_CHOICES][N_CONS])); 
  // double CCP[N_CHOICES][N_CONS];
  for (int i = 0; i < N_CHOICES; i++){
    for (int k = 0; k < N_CONS; k++){
      double ccp_ij = 0;
      for (int j = 0; j < N_SIMS; j++){
        ccp_ij += shares[i][j][k];
      } 
      // *(*(CCP+i)+j) =  ccp_ij/N_SIMS; 
      CCP[i][k] = ccp_ij/N_SIMS; 
      // printf("%f\n", CCP[i][j]);
    }
  }

  // Calculate the LL, loop over consumer's choices and CCP 
  double LL = 0;
  for (int j = 0; j < N_CONS; j++){
    LL -= log((CCP[(Y[j] - 1)][j]));
    // printf("%d\n", Y[j]);
    // printf("%f\n", CCP[(Y[j] - 1)][j]); 
    // printf("%f\n", log(CCP[(Y[j] - 1)][j])); 
  }
  clock_t toc = clock();
  double time_spent = (double)(toc - tic) / CLOCKS_PER_SEC;
  printf(" Time Elasped %f\n", time_spent);

  // free things from memory 
  free(X);
  free(W);
  free(S);
  free(Y);
  free(ExpUtils);
  free(shares);
  free(CCP);

 return 0;
}
