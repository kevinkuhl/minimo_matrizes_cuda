#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

// Kernel para encontrar o menor valor dado um vetor
__global__ void encontraMenor(int *vetor,int *menor, int tam)
{
        // Calcula a posicao inicial analisada
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        // Enquanto a posicao for menor do que o limite do vetor
        if (idx < tam)
        {
                // Calcula o elemento minimo, comparando com o valor em menor
                atomicMin(menor,vetor[idx]);
        }
}

// Funcao para abrir o arquivo e retornar o ponteiro
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
        // Define as variaveis a serem utilizadas
        FILE *file;
        int *mA_h,*mB_h;
        int *mA_d,*mB_d;

        // Dimensoes das matrizes a serem lidas
        int dimA[2],dimB[2];

        // Variaveis de iteracao
        int i,j;

        // Ponteiros para o resultado
        int *menorA_d;
        int *menorA_h;
        int *menorB_d;
        int *menorB_h;

        // Definindo as streams
        cudaStream_t stream1, stream2;

        // Criando as streams
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        // Abre o arquivo
        file = abrir_arquivo(argv[1]);

        // Le as dimensoes
        fscanf(file,"%d",&dimA[0]);
        fscanf(file,"%d",&dimA[1]);
        fscanf(file,"%d",&dimB[0]);
        fscanf(file,"%d",&dimB[1]);

        // Aloca espaco no host para as matrizes e para os resultados
        cudaMallocHost((void**)&mA_h,(dimA[0])*dimA[1]*(sizeof(int)));
        cudaMallocHost((void**)&mB_h,(dimB[0])*dimB[1]* (sizeof(int)));
        cudaMallocHost((void**)&menorA_h,sizeof(int));
        cudaMallocHost((void**)&menorB_h,sizeof(int));

        // Le as matrizes a partir do arquivo aberto
        for(i=0;i<dimA[0];i++)
                for(j=0;j<dimA[1];j++)
                        fscanf(file,"%d", &mA_h[i*dimA[1]+j]);

        for(i=0;i<dimB[0];i++)
                for(j=0;j<dimB[1];j++)
                        fscanf(file,"%d", &mB_h[i*dimB[1]+j]);
        
        // Fecha o arquivo
        fclose(file);

        // Aloca espaco no device para as matrizes e para os resultados
        cudaMalloc((void**)&mA_d,(dimA[0])*dimA[1]*(sizeof(int)));
        cudaMalloc((void**)&mB_d,(dimB[0])*dimB[1]* (sizeof(int)));
        cudaMalloc((void**)&menorA_d,sizeof(int));
        cudaMalloc((void**)&menorB_d,sizeof(int));

        // Inicializa o conteúdo do resultado no device com 10000
        cudaMemset(menorA_d,10000,sizeof(int));
        cudaMemset(menorB_d,10000,sizeof(int));
        
        // Inicializa as variaveis de thrads por bloco e de blocos por grid
        /* Aqui vale uma ressalva, como as matrizes podem ter dimensoes diferentes
        umas das outras, optamos por utilizar um bloco unico com tamanho total igual
        ao tamanho da matriz (produto das dimensoes).
        Isso evita termos que fazer dois loops for para iterar sobre os blocos (um para cada matriz)
        */
        int threadsPerBlockA = dimA[0]*dimA[1];
        int threadsPerBlockB = dimB[0]*dimB[1];
        int blocksPerGridA = ((dimA[0])+threadsPerBlockA-1)/threadsPerBlockA;
        int blocksPerGridB = ((dimB[0])+threadsPerBlockB-1)/threadsPerBlockB;

        // Copia asincronamente a memoria do host para o device
        cudaMemcpyAsync(mA_d,mA_h,(dimA[0]*dimA[1]*sizeof(int)), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(mB_d,mB_h,(dimB[0]*dimB[1]*sizeof(int)), cudaMemcpyHostToDevice, stream2);
        
        // Chama a funcao para encontrar o minimo na matriz A, utilizando a stream1
        encontraMenor <<<blocksPerGridA,threadsPerBlockA,0,stream1>>>(mA_d,menorA_d,dimA[0]*dimA[1]);
        // Copia o resultado para o host
        cudaMemcpy(menorA_h,menorA_d,sizeof(int), cudaMemcpyDeviceToHost);

        // Chama a funcao para encontrar o minimo na matriz B, utilizando a stream2
        encontraMenor <<<blocksPerGridB,threadsPerBlockB,0,stream2>>>(mB_d,menorB_d,dimB[0]*dimB[1]);
        // Copia o resultado para o host
        cudaMemcpy(menorB_h,menorB_d,sizeof(int), cudaMemcpyDeviceToHost);

        // Sincroniza as streams criadas
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        // Imprime os resultados
        printf("%d\n", *menorA_h);
        printf("%d", *menorB_h);

        // Libera o espaco alocado para as variaveis no host
        cudaFreeHost(menorA_h);
        cudaFreeHost(menorB_h);
        cudaFreeHost(mA_h);
        cudaFreeHost(mB_h);

        // Libera o espaco alocado para as variaveis no device
        cudaFree(mB_d);
        cudaFree(mA_d);
        cudaFree(menorA_d);
        cudaFree(menorB_d);

        // Libera o espaco alocado para as streams (fecha as streams)
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);

        // Sai do programa      
        exit(0);
}