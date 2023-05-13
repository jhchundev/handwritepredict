#define M_PI       3.14159265358979323846
#include "nn.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

// share function
//copy array
void copy(int m, int n, const float *x, float *y){
  int i, j;
  for (i = 0; i < m; i++)
  {
    for ( j = 0; j < n; j++)
    {
      y[i*n+j] = x[i*n+j];
    }
  }
}
// print array
void print(int m, int n, const float * x){
    for (int k = 0; k < m; k++)
    {
        for (int i = 0; i < n; i++)
        {
            printf("%f ", x[n*k+i]);
        }
        printf("\n");
    }
}
// get max index of array
int max_index(const float *y){
    int max_index=0;
    float max=0;
    for(int i=0;i<=9;i++){
        if(max<y[i]){
            max=y[i];
            max_index=i;
        }
    }  
    return max_index;
}

//補題2 y = Ax + bを計算する関数
void fc(int m,int n,const float*x,const float*A,const float*b,float*y){
    for(int i=0;i<m;i++){
        y[i]=b[i];
        for(int j=0;j<n;j++){
            y[i]+=A[i*n+j]*x[j];
        }
    }
}
// relu function
void relu(int n,const float*x,float*y){
    for (int i = 0; i < n; i++){
        if(x[i] <= 0)
            y[i] = 0;
        else
            y[i] = x[i];
    }
}
// relu function
void softmax(int n, const float * x, float * y){
  float x_max = 0;
  for (int r=0; r<n; r++){
    x_max = (x_max < x[r])?x[r]:x_max;
  }
  float sum_e = 0;
  for (int r=0; r<n; r++){
    sum_e += exp(x[r]-x_max);
  }
  for (int r=0; r<n; r++){
    y[r] = exp(x[r]-x_max)/sum_e;
  }
}
// softmax function
void softmax_bwd(int n, const float * y, unsigned char t, float * dEdx){
  for(int i=0;i<n;i++){
    dEdx[i]=0;
    dEdx[i] = (i==t)? y[i] - 1.0 : y[i];
  }
}
// get gradient of ReLU
void relu_bwd(int n, const float * x, const float * dEdy, float * dEdx){
  for(int i=0;i<n;i++){
    dEdx[i] = (x[i]>0)? dEdy[i] : 0;
  }
}
// get gradient of FC Layer
void fc_bwd(int m, int n, const float * x, const float * dEdy, const float * A, float * dEdA, float * dEdb, float * dEdx){
    
    int i=0;
    int j=0;
    //式14
    for (i = 0; i < m; i++)
       for (j = 0; j < n; j++)
           dEdA[i * n + j] = dEdy[i] * x[j];
    //式15
    for (i = 0; i < m; i++)
       dEdb[i] = dEdy[i];
    //式16
    for (i = 0; i < n; i++){
        dEdx[i]=0;
       for (j = 0; j < m; j++)
           dEdx[i] += A[j * n + i] * dEdy[j];
    }
}
// suffle of index array
void shuffle(int n,int *x){
    for (int i = 0; i < n; i++){
        int j =  (int)(rand() * (n + 1.0) / (1.0 + RAND_MAX));
        int temp = x[i];
        x[i] = x[j];
        x[j] = temp;
    }
}
//get entropy error。add 1e+7 to avoid log(0)
float cross_entropy_error(const float *y,int t){
  return -1*log(y[t]+1e-7);
}
//add array
void add(int n,const float *x,float *o){
  for (int i = 0; i < n;i++){
    o[i] = x[i] + o[i];
  }
}
// scale array
void scale(int n,float x,float *o){
  for (int i = 0; i < n;i++){
    o[i] = o[i] * x;
  }
}
//oの要素をxに初期化 
void init(int n,float x,float *o){
  for (int i = 0; i < n;i++){
    o[i] = x;
  }
}
// init array in [-1:1]
void rand_init(int n, float *o){
    int i;
    for (i = 0; i < n; i++){
        o[i] = ((float)rand() / ((float)RAND_MAX + 1)) * 2 - 1 ;
    }
}

// get random number in [0:1]
//https://omitakahiro.github.io/random/random_variables_generation.html#Exponential を参考に作成
float uniform( void ){
    return rand()/(RAND_MAX+1.0);
}

// normalization of normal distribution
void rand_init_normal(int n, float *o,float mu,float sigma){
    int i;
    for(i=0;i<n;i++){
        float z =sqrt( -2.0*log(uniform()) ) * cos( 2.0*M_PI*uniform() );
        o[i] =  sigma * z + mu;
    }
}
// save parameters of 1 layer
void save(const char * filename, int m, int n, const float * A, const float * b){
    FILE *fp;
    fp = fopen(filename,"w");

    fwrite(A, sizeof(float), m*n, fp);
    fwrite(b, sizeof(float), m, fp);

    fclose(fp);
}
// load parameters of 1 layer
void load(const char * filename, int m, int n, float * A, float * b){
    FILE *fp;
    fp = fopen(filename,"r");

    fread(A, sizeof(float), m*n, fp);
    fread(b, sizeof(float), m, fp);

    fclose(fp);
}

// inference of 3 layers
int inference3(const float*A,const float*b,const float*x,float *y){
    
    fc(10,784,x,A,b,y);
    relu(10,y,y);
    softmax(10,y,y);

    return max_index(y);
}
// get backwards of 3 layers
// FC -> RELU -> Softmax、save parameters of each layer and load to next layer
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dEdA, float * dEdb) {
  float *c_fc = malloc(sizeof(float)*784);
  float *c_relu = malloc(sizeof(float)*10);
  float *y_relu_10 = malloc(sizeof(float)*10);
  float *y_fc_784 = malloc(sizeof(float)*784);

  copy(1, 784, x, c_fc);
  fc(10,784,x,A,b,y);
  copy(1, 10, y, c_relu);
  relu(10, y, y);
  softmax(10,y,y); 

  softmax_bwd(10,y,t,y_relu_10);
  relu_bwd(10,c_relu,y_relu_10,y_fc_784);
  fc_bwd(10,784,c_fc,y_relu_10,A,dEdA,dEdb,y_fc_784);

  free(c_fc); free(c_relu); free(y_fc_784);  free(y_relu_10);
}
// main function of 3 layers
void main_3_layers(int train_count,int test_count,float * train_x,unsigned char * train_y,float * test_x, unsigned char * test_y){
    printf("N_3layers\n");
    // take memory
    float *y = malloc(sizeof(float)*10);
    float *dEdA=malloc(sizeof(float)*784*10);
    float *dEdb=malloc(sizeof(float)*10);
    float *A=malloc(sizeof(float)*784*10);
    float *b=malloc(sizeof(float)*10);
    int *index=malloc(sizeof(int)*train_count);

    float *dEdA_average=malloc(sizeof(float)*784*10); // gradient
    float *dEdb_average=malloc(sizeof(float)*10);  // gradient
    float *accuracy=malloc(sizeof(float)*10);
    // get condition
    int epoch;
    printf("Input epoch(integer): ");
    scanf("%d", &epoch);
    int n;
    printf("Input batch size(integer): ");
    scanf("%d", &n);
    float eta; //learning rate　update coefficient by average gradient
    printf("Input learning rate: ");
    scanf("%e", &eta);
    printf("Condition\n");
    printf("Epoch : %d , Batch size : %d, Learning rate : %f\n", epoch, n, eta);
    printf("Training Start\n");
    
    int N=train_count;
    // init of A,b
    srand(time(NULL));
    rand_init(784*10,A);
    rand_init(10,b);
    float max_accuracy=0;
    // loop for epoch
    for(int m=0;m<epoch;m++){
        printf("----------------------------\n");
        printf("Epoch %d/%d\n",m+1,epoch);
        //create index　　5-a
        for(int i=0;i<N ; i++) { 
            index[i] = i;
        }
        shuffle(N,index);

        //mini batch　5-b
        for(int i=0;i<N/n;i++){
            //init average gradient
            init(784*10,0,dEdA_average);
            init(10,0,dEdb_average);
            
            //calculate average gradient
            for(int j=0;j<n;j++){
                backward3(A,b,train_x+784*index[i*n+j],train_y[index[i*n+j]],y,dEdA,dEdb);
                add(784*10,dEdA,dEdA_average);
                add(10,dEdb,dEdb_average);
            }
            
            scale(784*10,-eta/n,dEdA_average);
            scale(10,-eta/n,dEdb_average);
            
            // updating A, b
            add(784*10,dEdA_average,A);
            add(10,dEdb_average,b);
            
            //printing progress
            fprintf(stderr, "\r[%3d/100]", 100*n*(i+1)/N);
            
        }

        float loss_train=0; //loss
        int sum_train=0;  //accuracy

        for(int k=0 ; k<train_count ; k++){
            if(inference3(A,b, train_x + k*784,y) == train_y[k]) { 
                sum_train++;  //sum up if correct
            }
            loss_train+=cross_entropy_error(y,train_y[k]);  //loss sum
        }
        putchar('\n');
        printf("Accuracy:%.2f\n",sum_train*100.0/train_count);
        printf("Loss :%.2f\n",loss_train/train_count);

        // テストデータに対する損失関数と正解率の計算と表示
        float loss_test=0; //loss
        int sum_test=0;  //accuracy

        for(int k=0 ; k<test_count ; k++){
            if(inference3(A,b, test_x + k*784,y) == test_y[k]) { 
                sum_test++;  //sum up if correct
            }
            loss_test+=cross_entropy_error(y,test_y[k]);  //loss sum
        }
        accuracy[m]=sum_test*100.0/test_count;
         
        if(max_accuracy<accuracy[m]){
            max_accuracy=accuracy[m];
        }
        printf("Accuracy(test):%.3f (%.3f)\n",accuracy[m],accuracy[m]-accuracy[m-1]);
        printf("Loss :%.2f\n",loss_test/test_count);     
    }
    printf("\nMax accuracy: %.3f\n",max_accuracy); 
    fprintf(stderr, "finish!\n");
    save("test.dat",10, 784, A, b);
    printf("Saved(test.dat)\n");
    //free memory
    free(A);free(b);free(y);free(index);free(dEdA);free(dEdb);
    free(dEdA_average);free(dEdb_average);

    printf("--------\n");
}
// 3 layers inference by Gaussian distribution
void normal_3_layers(int train_count,int test_count,float * train_x,unsigned char * train_y,float * test_x, unsigned char * test_y){
    printf("N_3layers\n");
    //take memory
    float *y = malloc(sizeof(float)*10);
    float *dEdA=malloc(sizeof(float)*784*10);
    float *dEdb=malloc(sizeof(float)*10);
    float *A=malloc(sizeof(float)*784*10);
    float *b=malloc(sizeof(float)*10);
    int *index=malloc(sizeof(int)*train_count);

    float *dEdA_average=malloc(sizeof(float)*784*10); //average gradient
    float *dEdb_average=malloc(sizeof(float)*10);  //average gradient
    float *accuracy=malloc(sizeof(float)*10);
    // get condition
    int epoch;
    printf("Input epoch(integer): ");
    scanf("%d", &epoch);
    int n;
    printf("Input batch size(integer): ");
    scanf("%d", &n);
    float eta; //learning rate　update coefficient by average gradient
    printf("Input learning rate: ");
    scanf("%e", &eta);
    printf("Condition\n");
    printf("Epoch : %d , Batch size : %d, Learning rate : %f\n", epoch, n, eta);
    printf("Training Start\n");
    
    int N=train_count;
    //init A,b
    srand(time(NULL));
    rand_init_normal(784*10,A, 0, sqrt(2.0/784));
    rand_init_normal(10,b, 0, sqrt(2.0/10));
    float max_accuracy=0;
    //loop for epoch
    for(int m=0;m<epoch;m++){
        printf("----------------------------\n");
        printf("Epoch %d/%d\n",m+1,epoch);
        //create index
        for(int i=0;i<N ; i++) { 
            index[i] = i;
        }
        shuffle(N,index);

        //mini batch　5-b
        for(int i=0;i<N/n;i++){
            // init average gradient
            init(784*10,0,dEdA_average);
            init(10,0,dEdb_average);
            
            //calculate average gradient
            for(int j=0;j<n;j++){
                backward3(A,b,train_x+784*index[i*n+j],train_y[index[i*n+j]],y,dEdA,dEdb);
                add(784*10,dEdA,dEdA_average);
                add(10,dEdb,dEdb_average);
            }
            
            scale(784*10,-eta/n,dEdA_average);
            scale(10,-eta/n,dEdb_average);
            
            // updating A, b
            add(784*10,dEdA_average,A);
            add(10,dEdb_average,b);
            
            //print progress
            fprintf(stderr, "\r[%3d/100]", 100*n*(i+1)/N);
            
        }
   
        float loss_train=0; //loss
        int sum_train=0;  //accuracy

        for(int k=0 ; k<train_count ; k++){
            if(inference3(A,b, train_x + k*784,y) == train_y[k]) { 
                sum_train++;  //add sum if correct
            }
            loss_train+=cross_entropy_error(y,train_y[k]);  //loss sum
        }
        putchar('\n');
        printf("Accuracy:%.2f\n",sum_train*100.0/train_count);
        printf("Loss :%.2f\n",loss_train/train_count);

        float loss_test=0; //loss
        int sum_test=0;  //accuracy

        for(int k=0 ; k<test_count ; k++){
            if(inference3(A,b, test_x + k*784,y) == test_y[k]) { 
                sum_test++;  //sum up if correct
            }
            loss_test+=cross_entropy_error(y,test_y[k]);  //sum loss
        }
        accuracy[m]=sum_test*100.0/test_count;
         
        if(max_accuracy<accuracy[m]){
            max_accuracy=accuracy[m];
        }
        printf("Accuracy(test):%.3f (%.3f)\n",accuracy[m],accuracy[m]-accuracy[m-1]);
        printf("Loss :%.2f\n",loss_test/test_count);     
    }
    printf("\nMax accuracy: %.3f\n",max_accuracy); 
    fprintf(stderr, "finish!\n");
    save("test.dat",10, 784, A, b);
    printf("Saved(test.dat)\n");
    //free memory
    free(A);free(b);free(y);free(index);free(dEdA);free(dEdb);
    free(dEdA_average);free(dEdb_average);

    printf("--------\n");
}
//inference of 6 layers
int inference6(const float*A1,const float*A2,const float *A3,const float*b1,const float *b2,const float *b3,const float*x,float *y){
    //take memory
    float *y1=malloc(sizeof(float)*50);
    float *y2=malloc(sizeof(float)*100);
    fc(50,784,x,A1,b1,y1);
    relu(50,y1,y1);
    fc(100,50,y1,A2,b2,y2);
    relu(100,y2,y2);
    fc(10,100,y2,A3,b3,y);
    softmax(10,y,y);
    
    //free memory
    free(y1);
    free(y2);
    //return max index
    return max_index(y);
    
}
// get backwards of 6 layers
void backward6(const float * A1,const float *A2,const float *A3, const float * b1,const float *b2,const float *b3,
 const float * x, unsigned char t, float * y, float * dEdA1,float *dEdA2,float *dEdA3, float * dEdb1,float *dEdb2,float *dEdb3){
    //take memory
    float *c_fc1=malloc(sizeof(float)*784);
    float *c_fc2=malloc(sizeof(float)*50);
    float *c_fc3=malloc(sizeof(float)*100);
    float *c_relu1=malloc(sizeof(float)*50);
    float *c_relu2=malloc(sizeof(float)*100);
    float *y0 = malloc(sizeof(float) * 784);
    float *y_relu1_50 = malloc(sizeof(float) * 50);
    float *y_relu2_100 = malloc(sizeof(float) * 100);
    float *y_fc3_10 = malloc(sizeof(float) *10);
    
    copy(1,784,x,c_fc1);
    fc(50, 784, x, A1, b1, c_relu1);
    relu(50,c_relu1,c_fc2);
    fc(100,50,c_fc2,A2,b2,c_relu2);
    relu(100,c_relu2,c_fc3);
    fc(10,100,c_fc3,A3,b3,y);
    softmax(10,y,y);

    softmax_bwd(10, y, t, y_fc3_10);
    fc_bwd(10, 100, c_fc3, y_fc3_10, A3, dEdA3, dEdb3, y_relu2_100);
    relu_bwd(100, c_relu2, y_relu2_100, y_relu2_100);
    fc_bwd(100, 50, c_fc2, y_relu2_100, A2, dEdA2, dEdb2, y_relu1_50);
    relu_bwd(50, c_relu1, y_relu1_50, y_relu1_50);
    fc_bwd(50, 784, c_fc1, y_relu1_50, A1, dEdA1, dEdb1, y0);

    //free memory
    free(c_fc1);free(c_fc2);free(c_fc3);
    free(c_relu1);free(c_relu2);
    free(y0);free(y_relu1_50);free(y_relu2_100);free(y_fc3_10);
}
// 6 layers main inference function
void main_6_layers(int train_count,int test_count,float * train_x,unsigned char * train_y, float * test_x, unsigned char * test_y){
    printf("Main_6layers\n");
    //take memory
    //gradient
    float *dEdA1 = malloc(sizeof(float)*784*50);
    float *dEdA2 = malloc(sizeof(float)*50*100);
    float *dEdA3 = malloc(sizeof(float)*100*10);
    float *dEdb1 = malloc(sizeof(float)*50);
    float *dEdb2 = malloc(sizeof(float)*100);
    float *dEdb3 = malloc(sizeof(float)*10);
    //average gradient
    float *dEdA1_average = malloc(sizeof(float)  *784 * 50);
    float *dEdA2_average = malloc(sizeof(float)  *50 * 100);
    float *dEdA3_average = malloc(sizeof(float)  *100 * 10);
    float *dEdb1_average = malloc(sizeof(float)  *50);
    float *dEdb2_average = malloc(sizeof(float)  *100);
    float *dEdb3_average = malloc(sizeof(float)  *10);
    //coef
    float *A1 = malloc(sizeof(float) *784*50);
    float *A2 = malloc(sizeof(float) *50*100);
    float *A3 = malloc(sizeof(float) *100*10);
    float *b1 = malloc(sizeof(float)*50);
    float *b2 = malloc(sizeof(float)*100);
    float *b3 = malloc(sizeof(float)*10);
    float *y = malloc(sizeof(float)*10);
    
    int *index=malloc(sizeof(int)*train_count);
    float *accuracy=malloc(sizeof(float)*10);
    //get condition
    int epoch;
    printf("Input epoch(integer): ");
    scanf("%d", &epoch);
    int n;
    printf("Input batch size(integer): ");
    scanf("%d", &n);
    float eta; //learning rate
    printf("Input learning rate: ");
    scanf("%e", &eta);
    printf("Condition\n");
    printf("Epoch : %d , Batch size : %d, Learning rate : %f\n", epoch, n, eta);
    printf("Training Start\n");
    srand(time(NULL));
    rand_init(784*50,A1);rand_init(50*100,A2);rand_init(100*10,A3);
    rand_init(50,b1);rand_init(100,b2);rand_init(10,b3);
    
    float max_accuracy=0;
        
    //loop for epoch
    for(int m=0;m<epoch;m++){
        printf("----------------------------\n");
        printf("Epoch %d/%d\n",m+1,epoch);
        //インデックスの生成(補題12)　　5-a
        for(int i=0;i<train_count ; i++) { 
            index[i] = i;
        }
        shuffle(train_count,index);
        
        for(int i=0;i<train_count/n;i++){
            // init average gradient
            init(784*50,0,dEdA1_average);
            init(50*100,0,dEdA2_average);
            init(100*10,0,dEdA3_average);
            init(50,0,dEdb1_average);
            init(100,0,dEdb2_average);
            init(10,0,dEdb3_average);
            
            // calculate average gradient
            for(int j=0;j<n;j++){
                backward6(A1,A2,A3,b1,b2,b3,train_x+784*index[i*n+j],train_y[index[i*n+j]],y,dEdA1,dEdA2,dEdA3,dEdb1,dEdb2,dEdb3); 
                add(784*50,dEdA1,dEdA1_average);
                add(50*100,dEdA2,dEdA2_average);
                add(100*10,dEdA3,dEdA3_average);
                add(50,dEdb1,dEdb1_average);
                add(100,dEdb2,dEdb2_average);
                add(10,dEdb3,dEdb3_average);

                }
            // learning rate update
            if(i>=train_count/n*2){
                eta=eta/2;
            }else if(i>=train_count*3/n*4){
                eta=eta/4;
            }
    
            scale(784*50,-eta/n,dEdA1_average);
            scale(50*100,-eta/n,dEdA2_average);
            scale(100*10,-eta/n,dEdA3_average);
            scale(50,-eta/n,dEdb1_average);
            scale(100,-eta/n,dEdb2_average);
            scale(10,-eta/n,dEdb3_average);
            
            //  update A, b
            add(784*50,dEdA1_average,A1);
            add(50*100,dEdA2_average,A2);
            add(100*10,dEdA3_average,A3);
            add(50,dEdb1_average,b1);
            add(100,dEdb2_average,b2);
            add(10,dEdb3_average,b3);

            //print loss
            fprintf(stderr, "\r[%3d/100]", 100*n*(i+1)/train_count);
            
        }
        
        float loss_train=0; //loss
        int sum_train=0;  //accuracy

        for(int k=0 ; k<train_count ; k++){
            if(inference6(A1,A2,A3,b1,b2,b3, train_x + k*784,y) == train_y[k]) { 
                sum_train++;  //sum up if correct
            }
            loss_train+=cross_entropy_error(y,train_y[k]);  //sum loss
        }
        putchar('\n');
        printf("Accuracy(train):%.3f\n",sum_train*100.0/train_count);
        printf("Loss(train) :%.3f\n",loss_train/train_count);

        float loss_test=0; //loss
        int sum_test=0;  //accuracy
        
        for(int k=0 ; k<test_count ; k++){
            if(inference6(A1,A2,A3,b1,b2,b3, test_x + k*784,y) == test_y[k]) { 
                sum_test++;  //sum up if correct
            }
            loss_test+=cross_entropy_error(y,test_y[k]);  //loss sum
        }
        accuracy[m]=sum_test*100.0/test_count;
         
        if(max_accuracy<accuracy[m]){
            max_accuracy=accuracy[m];
        }
        printf("Accuracy(test):%.3f (%.3f)\n",accuracy[m],accuracy[m]-accuracy[m-1]);
        printf("Loss(test):%.3f\n",loss_test/test_count);
           
    }
    printf("\n Max accuracy: %.3f\n",max_accuracy); 
    fprintf(stderr, "finish!\n"); 
    // save parameters
    save("6-fc1.dat", 50, 784, A1, b1);
    save("6-fc2.dat", 100, 50, A2, b2);
    save("6-fc3.dat", 10, 100, A3, b3);
    printf("Saved (6-fc.dat)\n");

    // free memory
    free(A1);free(A2);free(A3);free(b1);free(b2);free(b3);free(dEdA1);free(dEdA2);free(dEdA3);free(dEdb1);free(dEdb2);free(dEdb3);
    free(dEdA1_average);free(dEdA2_average);free(dEdA3_average);free(dEdb1_average);free(dEdb2_average);free(dEdb3_average);free(y);free(index);
}
//6 layers inference of gaussian distribution
void normal_6_layers(int train_count,int test_count,float * train_x,unsigned char * train_y, float * test_x, unsigned char * test_y){
    printf("N_6layers\n");
    //get memory
    //gradient
    float *dEdA1 = malloc(sizeof(float)*784*50);
    float *dEdA2 = malloc(sizeof(float)*50*100);
    float *dEdA3 = malloc(sizeof(float)*100*10);
    float *dEdb1 = malloc(sizeof(float)*50);
    float *dEdb2 = malloc(sizeof(float)*100);
    float *dEdb3 = malloc(sizeof(float)*10);
    //average gradient
    float *dEdA1_average = malloc(sizeof(float)  *784 * 50);
    float *dEdA2_average = malloc(sizeof(float)  *50 * 100);
    float *dEdA3_average = malloc(sizeof(float)  *100 * 10);
    float *dEdb1_average = malloc(sizeof(float)  *50);
    float *dEdb2_average = malloc(sizeof(float)  *100);
    float *dEdb3_average = malloc(sizeof(float)  *10);
    // coefficient
    float *A1 = malloc(sizeof(float) *784*50);
    float *A2 = malloc(sizeof(float) *50*100);
    float *A3 = malloc(sizeof(float) *100*10);
    float *b1 = malloc(sizeof(float)*50);
    float *b2 = malloc(sizeof(float)*100);
    float *b3 = malloc(sizeof(float)*10);
    float *y = malloc(sizeof(float)*10);
    
    int *index=malloc(sizeof(int)*train_count);
    float *accuracy=malloc(sizeof(float)*10);
   
    int epoch;
    printf("Input epoch(integer): ");
    scanf("%d", &epoch);
    int n;
    printf("Input batch size(integer): ");
    scanf("%d", &n);
    float eta; //eta　updating by average gradient
    printf("Input learning rate: ");
    scanf("%e", &eta);
    printf("Condition\n");
    printf("Epoch : %d , Batch size : %d, Learning rate : %f\n", epoch, n, eta);
    printf("Training Start\n");
    srand(time(NULL));
    rand_init_normal(784*50,A1,0,sqrt(2.0/784));
    rand_init_normal(50*100,A2,0,sqrt(2.0/50));
    rand_init_normal(100*10,A3,0,sqrt(2.0/100));

    rand_init_normal(50,b1,0,sqrt(2.0/50));
    rand_init_normal(100,b2,0,sqrt(2.0/100));
    rand_init_normal(10,b3,0,sqrt(2.0/10));
    
    float max_accuracy=0;
        
    // loop for each epoch
    for(int m=0;m<epoch;m++){
        printf("----------------------------\n");
        printf("Epoch %d/%d\n",m+1,epoch);
        //create index　　5-a
        for(int i=0;i<train_count ; i++) { 
            index[i] = i;
        }
        shuffle(train_count,index);
        
        //mini batch　5-b
        for(int i=0;i<train_count/n;i++){
            //init average gradient 0
            init(784*50,0,dEdA1_average);
            init(50*100,0,dEdA2_average);
            init(100*10,0,dEdA3_average);
            init(50,0,dEdb1_average);
            init(100,0,dEdb2_average);
            init(10,0,dEdb3_average);
            
            //calculate average gradient
            for(int j=0;j<n;j++){
                backward6(A1,A2,A3,b1,b2,b3,train_x+784*index[i*n+j],train_y[index[i*n+j]],y,dEdA1,dEdA2,dEdA3,dEdb1,dEdb2,dEdb3); 
                add(784*50,dEdA1,dEdA1_average);
                add(50*100,dEdA2,dEdA2_average);
                add(100*10,dEdA3,dEdA3_average);
                add(50,dEdb1,dEdb1_average);
                add(100,dEdb2,dEdb2_average);
                add(10,dEdb3,dEdb3_average);

                }
            //calculate learning rate
            if(i>=train_count/n*2){
                eta=eta/2;
            }else if(i>=train_count*3/n*4){
                eta=eta/4;
            }
            scale(784*50,-eta/n,dEdA1_average);
            scale(50*100,-eta/n,dEdA2_average);
            scale(100*10,-eta/n,dEdA3_average);
            scale(50,-eta/n,dEdb1_average);
            scale(100,-eta/n,dEdb2_average);
            scale(10,-eta/n,dEdb3_average);
            
            //  update A, b
            add(784*50,dEdA1_average,A1);
            add(50*100,dEdA2_average,A2);
            add(100*10,dEdA3_average,A3);
            add(50,dEdb1_average,b1);
            add(100,dEdb2_average,b2);
            add(10,dEdb3_average,b3);

            //print progress
            fprintf(stderr, "\r[%3d/100]", 100*n*(i+1)/train_count);
            
        }
        
        float loss_train=0; //loss
        int sum_train=0;  //accuracy

        for(int k=0 ; k<train_count ; k++){
            if(inference6(A1,A2,A3,b1,b2,b3, train_x + k*784,y) == train_y[k]) { 
                sum_train++;  //sum if correct
            }
            loss_train+=cross_entropy_error(y,train_y[k]);  //sum loss
        }
        putchar('\n');
        printf("Accuracy(train):%.3f\n",sum_train*100.0/train_count);
        printf("Loss(train) :%.3f\n",loss_train/train_count);

        float loss_test=0; //loss
        int sum_test=0;  //correct number
        
        for(int k=0 ; k<test_count ; k++){
            if(inference6(A1,A2,A3,b1,b2,b3, test_x + k*784,y) == test_y[k]) { 
                sum_test++;  // add if correct
            }
            loss_test+=cross_entropy_error(y,test_y[k]);  // adding loss
        }
        accuracy[m]=sum_test*100.0/test_count;
         
        if(max_accuracy<accuracy[m]){
            max_accuracy=accuracy[m];
        }
        printf("Accuracy(test):%.3f (%.3f)\n",accuracy[m],accuracy[m]-accuracy[m-1]);
        printf("Loss(test):%.3f\n",loss_test/test_count);
           
    }
    printf("\n Max accuracy: %.3f\n",max_accuracy); 
    fprintf(stderr, "finish!\n"); 
    // save parameter
    save("6-fc1.dat", 50, 784, A1, b1);
    save("6-fc2.dat", 100, 50, A2, b2);
    save("6-fc3.dat", 10, 100, A3, b3);
    printf("Saved (6-fc.dat)\n");

    // free memory
    free(A1);free(A2);free(A3);free(b1);free(b2);free(b3);free(dEdA1);free(dEdA2);free(dEdA3);free(dEdb1);free(dEdb2);free(dEdb3);
    free(dEdA1_average);free(dEdA2_average);free(dEdA3_average);free(dEdb1_average);free(dEdb2_average);free(dEdb3_average);free(y);free(index);
}
//main function
int main(){
  
  float *train_x = NULL;
  unsigned char *train_y = NULL;
  int train_count = -1;

  float *test_x = NULL;
  unsigned char *test_y = NULL;
  int test_count = -1;

  int width = -1;
  int height = -1;

  load_mnist(&train_x, &train_y, &train_count,
             &test_x, &test_y, &test_count,
             &width, &height);

#if 0
  volatile float x = 0;
  volatile float y = 0;
  volatile float z = x/y;
#endif

  printf("NN Loaded OK.\n");
  //get memory
  float *A=malloc(sizeof(float)*784*10);
  float *b=malloc(sizeof(float)*10);
  float *A1 = malloc(sizeof(float) *784*50);
  float *A2 = malloc(sizeof(float) *50*100);
  float *A3 = malloc(sizeof(float) *100*10);
  float *b1 = malloc(sizeof(float)*50);
  float *b2 = malloc(sizeof(float)*100);
  float *b3 = malloc(sizeof(float)*10);
  float *y = malloc(sizeof(float)*10);

  char filename[20], filename2[20];

  int mode, opt;

  while(1)
  {
    printf("Mode 1 : Learning / Mode 2 : Quiz / Mode 3 : Quit\n");
    printf("Input mode : ");
    scanf("%d", &mode);

    if (mode == 1)
    {
        printf("Learning mode\n");
        printf("3 layer (1)/ normal 3 layer (2)\n6 layer (3)/ normal 6 layer (4)\n");
        printf("Choose optimizer : ");
        scanf("%d", &opt);

        switch (opt)
        {
        case 1:
        main_3_layers(train_count, test_count, train_x, train_y, test_x, test_y);
        break;

        case 2:
        normal_3_layers(train_count, test_count, train_x, train_y, test_x, test_y);
        break;

        case 3:
        main_6_layers(train_count,test_count, train_x, train_y, test_x, test_y);  
        break;

        case 4:
        normal_6_layers(train_count,test_count, train_x, train_y, test_x, test_y);  
        break;   

        default:
        break;
        }

    }else if(mode == 2){
        printf("Quiz mode\n");
        printf("Choose Inference Model 3-layers(1) / 6-layers(2)\n");
        int quiz;
        printf("Model : ");
        scanf("%d", &quiz);

        switch (quiz)
        {
        case 1:
            

            printf("Load test.dat\n");
            load("test.dat", 10, 784, A, b);
            printf("Loaded\n");

            printf("Input Filename : ");
            scanf("%s", filename);

        
            float *x1 = load_mnist_bmp(filename);
            if (x1 != 0)
            {
            printf("Test File Loaded!\n");
            }else{
                printf("Test File Loaded Failed, Please Retry\n");
            }
            
            int answer = inference3(A, b, x1, y );

            printf("Answer : %d\n", answer);
            free(A); free(b); free(y);
            break;
            
        case 2:
            
            load("6-fc1.dat", 50, 784, A1, b1);
            load("6-fc2.dat", 100, 50, A2, b2);
            load("6-fc3.dat", 10, 100, A3, b3);
            printf("Loaded\n");

            printf("Input Filename : ");
            scanf("%s", filename2);

            float *x2 = load_mnist_bmp(filename2);

            if (x2 != 0)
            {
            printf("Test File Loaded!\n");
            }else{
                printf("Test File Loaded Failed, Please Retry\n");
            }
            int answer2 = inference6(A1, A2, A3,b1, b2, b3, x2, y);

            printf("Answer : %d\n", answer2);
            printf("----------------------------\n");

        default:
            break;
        }
        
    }else{
        free(A); free(b); free(A1); free(A2); free(A3);
        free(b1); free(b2); free(b3); free(y);
        return 0;
    }

  }
  
}
