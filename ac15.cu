#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void encontraMenor(int *mG_glb,int *menor, int tam)
{
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        if (idx < tam)
        {
                atomicMin(menor,mG_glb[idx]);
        }
}

FILE *abrir_arquivo(char nome[])
{
        FILE *pFile;
        pFile = fopen(nome, "r");
        if(pFile==NULL)
                exit(0);
        return pFile;
}

int main(int argc,char **argv)
{
        FILE *file;
        int *mA_h,*mB_h;
        int *mA_d,*mB_d;
        int dimA[2],dimB[2];
        int i,j;
        int *menorA_d;
        int *menorA_h;
        int *menorB_d;
        int *menorB_h;
        cudaStream_t stream1, stream2;

        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        //abri o arquivo das matrizes
        file = abrir_arquivo(argv[1]);
        //leitura das dimenssões
        fscanf(file,"%d",&dimA[0]);
        fscanf(file,"%d",&dimA[1]);
        fscanf(file,"%d",&dimB[0]);
        fscanf(file,"%d",&dimB[1]);
        //printf("%d %d \n %d %d\n" ,dimA[0], dimA[1], dimB[0], dimB[1]);
        //alocando matrizes e retorno
        //mA_h=(int* )malloc(dimA[0]*dimA[1]*sizeof(int));
        //mB_h=(int* )malloc(dimB[0]*dimB[1]*sizeof(int));
        //menor_h=(int *) malloc(1*sizeof(int));

        cudaMallocHost((void**)&mA_h,(dimA[0])*dimA[1]*(sizeof(int)));
        cudaMallocHost((void**)&mB_h,(dimB[0])*dimB[1]* (sizeof(int)));
        menorA_h=(int *) malloc(1*sizeof(int));
        menorB_h=(int *) malloc(1*sizeof(int));
        //cudaMallocHost((void**)&menorA_h,sizeof(int));
        //cudaMallocHost((void**)&menorB_h,sizeof(int));



        //Leitura do arquivo para matriz A e B
        for(i=0;i<dimA[0];i++)
                for(j=0;j<dimA[1];j++)
                        fscanf(file,"%d", &mA_h[i*dimA[1]+j]);

        for(i=0;i<dimB[0];i++)
                for(j=0;j<dimB[1];j++)
                        fscanf(file,"%d", &mB_h[i*dimB[1]+j]);
        fclose(file);

        cudaMalloc((void**)&mA_d,(dimA[0])*dimA[1]*(sizeof(int)));
        cudaMalloc((void**)&mB_d,(dimB[0])*dimB[1]* (sizeof(int)));
        cudaMalloc((void**)&menorA_d,sizeof(int));
        cudaMalloc((void**)&menorB_d,sizeof(int));

        //Inicializa o conteúdo da variável no device com 10000
        cudaMemset(menorA_d,10000,sizeof(int));
        cudaMemset(menorB_d,10000,sizeof(int));
        
        int threadsPerBlock = 2;
        int blocksPerGridA = ((dimA[0])+threadsPerBlock-1)/threadsPerBlock;
        int blocksPerGridB = ((dimB[0])+threadsPerBlock-1)/threadsPerBlock;

        //copia arquivos do host para o device
        cudaMemcpyAsync(mA_d,mA_h,(dimA[0]*dimA[1]*sizeof(int)), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(mB_d,mB_h,(dimB[0]*dimA[1]*sizeof(int)), cudaMemcpyHostToDevice, stream2);

        encontraMenor <<<blocksPerGridA,threadsPerBlock,0,stream1>>>(mA_d,&menorA_d,dimA[0]*dimA[1]);
        cudaMemcpy(menorA_h,menorA_d,sizeof(int), cudaMemcpyDeviceToHost);

        encontraMenor <<<blocksPerGridB,threadsPerBlock,0,stream2>>>(mB_d,&menorB_d,dimB[0]*dimB[1]);
        cudaMemcpy(menorB_h,menorB_d,sizeof(int), cudaMemcpyDeviceToHost);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        printf("Menor valor da matriz A: %d\n", menorA_h);
        printf("Menor valor da matriz B: %d\n", menorB_h);

        printf("Matriz A - Flatten\n");
        for(i=0;i<dimA[0];i++)
                for(j=0;j<dimA[1];j++)
                        printf("%d ", mA_h[i*dimA[1]+j]);
                printf("\n\n");
        printf("Matriz B - Flatten\n");
        for(i=0;i<dimB[0];i++)
                for(j=0;j<dimB[1];j++)
                        printf("%d ", mB_h[i*dimB[1]+j]);
        printf("\n");

        cudaFreeHost(menorA_h);
        cudaFreeHost(menorB_h);
        cudaFreeHost(mA_h);
        cudaFreeHost(mB_h);

        cudaFree(mB_d);
        cudaFree(mA_d);
        cudaFree(menorA_d);
        cudaFree(menorB_d);

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);

        //chama a função com o stream
        //encontraMenor <<<BPG,TPB>>> (mA_d,menor_d,dimA[0]*dimA[1]);
        //
        //Copia o resultado da soma de volta para o host
        //cudaMemcpy(menor_h,menor_d,sizeof(int), cudaMemcpyDeviceToHost);
        //sincroniza os valores
        //cudaStreamSynchronize(stream);
        //printf("Menor elemento da matriz A\n");
        //printf("%d\n\n",*menor_h);
        //imprime
        //free(menor_h);
        //free(mA_h);
        //free(mB_h);
        //cudaFree(menor_d);
        //Desaloca os vetores no device
        //cudaFree(mB_d);
        //cudaFree(mA_d);
        exit(0);
}