/*
#include<iostream>
#include<stdio.h>
#include<string.h>
using namespace std;
void insert(float arr[],int n,int a){
    arr[n]=a;
}
void heapify(float arr[],int n,int i){
    int Min=i;
    int left=2*i+1;
    int right=2*i+2;
    if (left < n && arr[left] < arr[Min])
        Min= left;
    if (right < n && arr[right] < arr[Min])
        Min= right;
    if (Min != i) {
        float b=arr[i];
        arr[i]=arr[Min];
        arr[Min]=b;
        heapify(arr, n, Min);
    }
}
void heapSort(float arr[], int N)
{
    for (int i = N / 2 - 1; i >= 0; i--)
        heapify(arr, N, i);
    for (int i = N - 1; i >= 0; i--) {
        float b=arr[0];
        arr[0]=arr[i];
        arr[i]=b;
        heapify(arr, i, 0);
    }
}
void printArray(float arr[], int N)
{
    for (int i = 0; i < N; i++)
        cout<<arr[i]<<" ";
    cout<<"\n";
}
void knapsack(float arr1[], float arr2[],int max,int n){
    float arr3[n];
    float arr4[n];
    float sum=0;
    int size=0;
    for(int i=0;i<n;i++){
        arr3[i]=arr1[i]/arr2[i];
    }
    for(int i=0;i<n;i++){
        arr4[i]=arr3[i];
    }
    heapSort(arr3,10);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(arr3[i]==arr4[j]){
                int tempsize=size+arr2[j];
                if(tempsize<max){
                    sum+=arr1[j];
                    size+=arr2[j];
                }
                else{
                    while(size<max){
                        sum+=(arr1[j]/arr2[j]);
                        size+=1;
                    }
                }
            }
        }
    }
    cout<<"Total price possible= "<<sum;
}
int main(){
    cout<<"21BCE3503-Naethen Luke\n";
    float arr1[] ={30,40,45,77,90,60,20,10,5,49};
    printArray(arr1,10);
    float arr2[]={5,10,15,22,25,15,10,3,5,7};
    printArray(arr2,10);
    knapsack(arr1,arr2,100,10);
}





#include <iostream>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
using namespace std;
struct MinHNode {
  unsigned freq;
  char item;
  struct MinHNode *left, *right;
};

struct MinH {
  unsigned size;
  unsigned capacity;
  struct MinHNode **array;
};
struct MinHNode *newNode(char item, unsigned freq) {
  struct MinHNode *temp = (struct MinHNode *)malloc(sizeof(struct MinHNode));

  temp->left = temp->right = NULL;
  temp->item = item;
  temp->freq = freq;

  return temp;
}
struct MinH *createMinH(unsigned capacity) {
  struct MinH *minHeap = (struct MinH *)malloc(sizeof(struct MinH));
  minHeap->size = 0;
  minHeap->capacity = capacity;
  minHeap->array = (struct MinHNode **)malloc(minHeap->capacity * sizeof(struct MinHNode *));
  return minHeap;
}
void printArray(int arr[], int n) {
  int i;
  for (i = 0; i < n; ++i)
    cout << arr[i];

  cout << "\n";
}
void swapMinHNode(struct MinHNode **a, struct MinHNode **b) {
  struct MinHNode *t = *a;
  *a = *b;
  *b = t;
}
void Heapify(struct MinH *minHeap, int x) {
  int min = x;
  int left = 2 * x + 1;
  int right = 2 * x + 2;

  if (left < minHeap->size && minHeap->array[left]->freq < minHeap->array[min]->freq)
    min = left;

  if (right < minHeap->size && minHeap->array[right]->freq < minHeap->array[min]->freq)
    min = right;

  if (min != x) {
    swapMinHNode(&minHeap->array[min],
           &minHeap->array[x]);
    Heapify(minHeap, min);
  }
}

int checkSizeOne(struct MinH *minHeap) {
  return (minHeap->size == 1);
}
struct MinHNode *extractMin(struct MinH *minHeap) {
  struct MinHNode *temp = minHeap->array[0];
  minHeap->array[0] = minHeap->array[minHeap->size - 1];

  --minHeap->size;
  Heapify(minHeap, 0);

  return temp;
}
void insertMinHeap(struct MinH *minHeap, struct MinHNode *minHeapNode) {
  ++minHeap->size;
  int i = minHeap->size - 1;

  while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
    minHeap->array[i] = minHeap->array[(i - 1) / 2];
    i = (i - 1) / 2;
  }

  minHeap->array[i] = minHeapNode;
}

void buildMinHeap(struct MinH *minHeap) {
  int n = minHeap->size - 1;
  int i;

  for (i = (n - 1) / 2; i >= 0; --i)
    Heapify(minHeap, i);
}

int isLeaf(struct MinHNode *root) {
  return !(root->left) && !(root->right);
}

struct MinH *createAndBuildMinHeap(char item[], int freq[], int size) {
  struct MinH *minHeap = createMinH(size);

  for (int i = 0; i < size; ++i)
    minHeap->array[i] = newNode(item[i], freq[i]);

  minHeap->size = size;
  buildMinHeap(minHeap);

  return minHeap;
}

struct MinHNode *buildHfTree(char item[], int freq[], int size) {
  struct MinHNode *left, *right, *top;
  struct MinH *minHeap = createAndBuildMinHeap(item, freq, size);

  while (!checkSizeOne(minHeap)) {
    left = extractMin(minHeap);
    right = extractMin(minHeap);

    top = newNode('$', left->freq + right->freq);

    top->left = left;
    top->right = right;

    insertMinHeap(minHeap, top);
  }
  return extractMin(minHeap);
}
void printHCodes(struct MinHNode *root, int arr[], int top) {
  if (root->left) {
    arr[top] = 0;
    printHCodes(root->left, arr, top + 1);
  }

  if (root->right) {
    arr[top] = 1;
    printHCodes(root->right, arr, top + 1);
  }
  if (isLeaf(root)) {
    cout << root->item << "  | ";
    printArray(arr, top);
  }
}

// Wrapper function
void HuffmanCodes(char item[], int freq[], int size) {
  struct MinHNode *root = buildHfTree(item, freq, size);

  int arr[50], top = 0;

  printHCodes(root, arr, top);
}

int main() {
	cout<<"21BCE3503-Naethen Luke\n";
  	char arr[] = {'A','B','C','D','E','F'};
  	int freq[] = {5,9,12,13,16,45};
  	int size = 6;
  	HuffmanCodes(arr, freq, size);
}



#include <stdio.h>  
#include <string.h>  
int i, j, a, b, Table[20][20];  
char S1[20] = "naethen", S2[20] = "kanichaatil";  
void LCS() {  
  a = strlen(S1);  
  b = strlen(S2);  
  for (i = 0; i <= a; i++)  
    Table[i][0] = 0;  
  for (i = 0; i <= b; i++)  
    Table[0][i] = 0;   
  for (i = 1; i <= a; i++)  
    for (j = 1; j <= b; j++) {  
      if (S1[i - 1] == S2[j - 1]) {  
        Table[i][j] = Table[i - 1][j - 1] + 1;  
      } else if (Table[i - 1][j] >= Table[i][j - 1]) {  
        Table[i][j] = Table[i - 1][j];  
      } else {  
        Table[i][j] = Table[i][j - 1];  
      }  
    }  
  
  int index = Table[a][b];  
  char LCS[index + 1];  
  LCS[index] = '\0';  
  
  int i = a, j = b;  
  while (i > 0 && j > 0) {  
    if (S1[i - 1] == S2[j - 1]) {  
      LCS[index - 1] = S1[i - 1];  
      i--;  
      j--;  
      index--;  
    }  
  
    else if (Table[i - 1][j] > Table[i][j - 1])  
      i--;  
    else  
      j--;  
  }  
  printf("S1 : %s \nS2 : %s \n", S1, S2);  
  printf("LCS: %s", LCS);  
}  
  
int main() {  
  LCS();  
  printf("\n");  
}  


#include<bits/stdc++.h>
using namespace std;
int MatrixChainOrder(int p[], int n){ 
  int m[n][n]; 
  int i, j, k, L, q; 
  for (i=1; i<n; i++) 
    m[i][i] = 0; 
  for (L=2; L<n; L++){ 
    for (i=1; i<n-L+1; i++){ 
      j = i+L-1; 
      m[i][j] = INT_MAX; 
      for (k=i; k<=j-1; k++){ 
        q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]; 
        if (q < m[i][j]) 
          m[i][j] = q; 
      } 
    } 
  } 
  return m[1][n-1]; 
} 

int main(){ 
  int arr[] = {5,4,6,2,7}; 
  int size = 5; 
    cout<<MatrixChainOrder(arr,size)<<" operations";

}



#include <limits.h>
#include <stdio.h>
int MatrixChainMult(int p[], int i, int j)
{
    if (i == j)
        return 0;
    int k;
    int min = 1000;
    int count;
    for (k = i; k < j; k++)
    {
		count = MatrixChainMult(p, i, k) + MatrixChainMult(p, k + 1, j) + p[i - 1] * p[k] * p[j];

        if (count < min)
            min = count;
    }
    return min;
}
int main(){
    int arr[] = { 4,3,5,6,1};
    int n = 5;
    printf("Minimum number of multiplications is %d ",MatrixChainMult(arr, 1, n - 1));
}


#include <stdio.h>
#define NUM_LINE 2
#define NUM_STATION 4
int min(int a, int b) { return a < b ? a : b; }
 
int carAssembly(int a[][NUM_STATION], int t[][NUM_STATION], int *e, int *x)
{
    int T1[NUM_STATION], T2[NUM_STATION], i;
 
    T1[0] = e[0] + a[0][0];
    T2[0] = e[1] + a[1][0]; 
    for (i = 1; i < NUM_STATION; ++i)
    {
        T1[i] = min(T1[i-1] + a[0][i], T2[i-1] + t[1][i] + a[0][i]);
        T2[i] = min(T2[i-1] + a[1][i], T1[i-1] + t[0][i] + a[1][i]);
    }
    return min(T1[NUM_STATION-1] + x[0], T2[NUM_STATION-1] + x[1]);
}
 
int main()
{
    int a[][NUM_STATION] = {{5,6,4,3},
                {2,3,1,10}};
    int t[][NUM_STATION] = {{0, 7, 8,4},
                {9,3,5,7}};
    int e[] = {10, 11}, x[] = {10, 7};
 
    printf("%d", carAssembly(a, t, e, x));
 
    return 0;
}






#include <stdio.h>
int maxSubarraySum(int arr[], int n) {
 int Sum = -1000;
 int start=0;
 int end=0;
 int i;
 for(i=0; i <= n - 1; i++) {
   int curr = 0;
   int j;
   for (j=i; j <= n - 1; j++) {
     curr += arr[j];
     if (curr > Sum) {
       Sum = curr;
       start=i;
       end=j;
     }
   }

 }
 printf("{");
 for(int i=start;i<end;i++){
 	printf(" %d, ",arr[i]);
 }
 printf("%d ",arr[end]);
 printf("}");
 printf("\n");
 return Sum;
 
}
int main() {
   int a[] = {1, 4, 5, -2, 10, -11, 5};
   printf("%d", maxSubarraySum(a, 7));
   return 0;
}




#include <stdio.h>
int is_attack(int i, int j, int board[6][6], int N) {
  int k, l;
  for(k=1; k<=i-1; k++) {
    if(board[k][j] == 1)
      return 1;
  }
  k = i-1;
  l = j+1;
  while (k>=1 && l<=N) {
    if (board[k][l] == 1)
      return 1;
    k=k+1;
    l=l+1;
  }
  k = i-1;
  l = j-1;
  while (k>=1 && l>=1) {
    if (board[k][l] == 1)
      return 1;
    k=k-1;
    l=l-1;
  }
  return 0;
}

int n_queen(int r, int n, int N, int board[6][6]) {
  if (n==0)
    return 1;
  int j;
  for (j=1; j<=N; j++) {
    if(!is_attack(r, j, board, N)) {
      board[r][j] = 1;

      if (n_queen(r+1, n-1, N, board))
        return 1;
      board[r][j] = 0;
    }
  }
  return 0;
}

int main() {
  int board[6][6];
  int i, j;
  for(i=0;i<=5;i++) {
    for(j=0;j<=5;j++)
      board[i][j] = 0;
  }
  n_queen(1, 5, 5, board);
  for(i=1;i<=5;i++) {
    for(j=1;j<=5;j++){
    	if(board[i][j]==0){
    		printf(" . ");
		}
		if(board[i][j]==1){
			printf(" Q ");
		}
	}
    printf("\n");
  }
  return 0;
}


#include <stdio.h>
void printMat(int mat[][4]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (mat[i][j] == 999)
        printf("%4s", "INF");
      else
        printf("%4d", mat[i][j]);
    }
    printf("\n");
  }
}

void floyd(int graph[][4]) {
  int mat[4][4], i, j, k;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      mat[i][j] = graph[i][j];
  for (k = 0; k < 4; k++) {
    for (i = 0; i < 4; i++) {
      for (j = 0; j < 4; j++) {
        if (mat[i][k] + mat[k][j] < mat[i][j])
          mat[i][j] = mat[i][k] + mat[k][j];
      }
    }
  }
  printMat(mat);
}


int main() {
  int graph[4][4] = {{0, 3, 999, 5},
             {2, 0, 999, 4},
             {999, 1, 0, 999},
             {999, 999, 2, 0}};
  floyd(graph);
}


#include <stdio.h>
#include <stdlib.h>
#include <limits.h>


struct Edge {
    int source;
    int destination;
    int weight;
};

struct Graph {
    int V;  
    int E;
    struct Edge edges[1000];
};

void bellman_ford(struct Graph* graph, int source) {
    int distances[1000];
    int i, j;
    
    for (i = 0; i < graph->V; ++i) {
        distances[i] = INT_MAX;
    }
    distances[source] = 0;
    for (i = 0; i < graph->V-1; ++i) {
        for (j = 0; j < graph->E; ++j) {
            int u = graph->edges[j].source;
            int v = graph->edges[j].destination;
            int weight = graph->edges[j].weight;
            if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
            }
        }
    }
    for (j = 0; j < graph->E; ++j) {
        int u = graph->edges[j].source;
        int v = graph->edges[j].destination;
        int weight = graph->edges[j].weight;
        if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
            printf("Graph contains a negative-weight cycle\n");
            return;
        }
    }
    printf("Vertex   Distance from source\n");
    for (i = 0; i < graph->V; ++i) {
        printf("%d \t\t %d\n", i, distances[i]);
    }
}

int main() {
    int V, E, i, source;
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    
    printf("Enter the number of vertices: ");
    scanf("%d", &V);
    
    printf("Enter the number of edges: ");
    scanf("%d", &E);
    
    graph->V = V;
    graph->E = E;
    
    for (i = 0; i < E; ++i) {
        printf("Enter source, destination and weight of edge %d: ", i+1);
        scanf("%d %d %d", &graph->edges[i].source, &graph->edges[i].destination, &graph->edges[i].weight);
    }
    
    printf("Enter the source node: ");
    scanf("%d", &source);
    
    bellman_ford(graph, source);
    
    return 0;
}




#include <stdio.h>

int n;
int e;
int size[1000][1000];
int f[1000][1000];
int color[1000];
int pred[1000];

int min(int x, int y) {
  return x < y ? x : y;
}

int head, tail;
int q[1000 + 2];

void enqueue(int x) {
  q[tail] = x;
  tail++;
  color[x] = 1;
}

int dequeue() {
  int x = q[head];
  head++;
  color[x] = 2;
  return x;
}
int bfs(int start, int target) {
  int i, j;
  for (i = 0; i < n; i++) {
    color[i] = 0;
  }
  head = tail = 0;
  enqueue(start);
  pred[start] = -1;
  while (head != tail) {
    i = dequeue();
    for (j = 0; j < n; j++) {
      if (color[j] == 0 && size[i][j] - f[i][j] > 0) {
        enqueue(j);
        pred[j] = i;
      }
    }
  }
  return color[target] == 2;
}
int Fulkerson(int source, int sink) {
  int i, j, k;
  int mf= 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      f[i][j] = 0;
    }
  }
  while (bfs(source, sink)) {
    int increment = 100000000;
    for (k = n - 1; pred[k] >= 0; k = pred[k]) {
      increment = min(increment, size[pred[k]][k] - f[pred[k]][k]);
    }
    for (k = n - 1; pred[k] >= 0; k = pred[k]) {
      f[pred[k]][k] += increment;
      f[k][pred[k]] -= increment;
    }
    mf+= increment;
  }
  return mf;
}

int main() {
	printf("21BCE3503- Naethen Luke\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      size[i][j] = 0;
    }
  }
  n = 6;
  e = 7;

  size[0][1] = 3;
  size[0][4] = 8;
  size[1][2] = 2;
  size[2][4] = 1;
  size[2][5] = 4;
  size[3][5] = 5;
  size[4][2] = 9;
  size[4][3] = 6;

  int s = 0, t = 5;
  printf("Max Flow: %d\n",Fulkerson(s, t));
}





#include<cstdio>
#include<queue>
#include<cstring>
#include<vector>
#include<iostream>
using namespace std;
int c[10][10];
int f[10][10];
vector<int> g[10];
int parList[10];
int Path[10];
int bfs(int s, int e)
{
   memset(parList, -1, sizeof(parList));
   memset(Path, 0, sizeof(Path));
   queue<int> q;
   q.push(s);
   parList[s] = -1;
   Path[s] = 999;
   while(!q.empty())
   {
      int curr = q.front();
      q.pop();
      for(int i=0; i<g[curr].size(); i++)
      {
         int to = g[curr][i];
         if(parList[to] == -1)
         {
            if(c[curr][to] - f[curr][to] > 0)
            {
               parList[to] = curr;
               Path[to] = min(Path[curr],
               c[curr][to] - f[curr][to]);
               if(to == e)
               {
                  return Path[e];
               }
               q.push(to);
            }
         }
      }
   }
   return 0;
}
int edmondsKarp(int s, int e)
{
   int mf = 0;
   while(true)
   {
      int f = bfs(s, e);
      if (f == 0)
      {
         break;
      }
      mf += f;
      int curr = e;
      while(curr != s)
      {
         int prevNode = parList[curr];
         k[prev][curr] += f;
         k[curr][prev] -= f;
         curr = prev;
      }
   }
return mf;
}
int main()
{
   int no, ed;
   cout<<"enter the number of nodes and edges\n";
   cin>>no>>ed;
   for(int e = 0; e < ed; e++)
   {
      cout<<"enter the start and end vertex along with capacity\n";
      int from, to, cap;
      cin>>from>>to>>cap;
      c[from][to] = cap;
      g[from].push_back(to);
      g[to].push_back(from);
   }
   int source, sink;
   cout<<"enter the source and sink\n";
   cin>>source>>sink;
   int mf = edmondsKarp(source, sink);
   cout<<endl<<endl<<"Max Flow is:"<<mf<<endl;
}




#include<stdio.h>
#include<string.h>
#include<stdlib.h>
int Naive(char a[], char b[], int n1, int n2){
	int f=0;
	for(int i=0;i<n1;i++){
		int j=0;
		if(a[i]==b[j]){
			char c[n2];
			int s=0;
			for(int k=i;k<i+n2;k++){
				c[s]=a[k];
				s++;
			}
			for(int l=0;l<n2;l++){
				if(c[l]==b[l]){
					f=i;
				}
				else{
					f=0;
					break;
				}
			}
		}
	}
	if(f==0){
		return(-1);
	}
	else{
		return(f);
	}
}
int main(){
	int n1;
	int n2;
	printf("Enter the size of main string-");
	scanf("%d",&n1);
	printf("Enter the main string-");
	char a[n1];
	scanf("%s",&a);
	printf("Enter the size of sub string-");
	scanf("%d",&n2);
	char b[n2];
	printf("Enter the sub string-");
	scanf("%s",&b);
	int k=Naive(a,b,n1,n2);
	if(k==-1){
		printf("String not matching");
	}
	else{
		printf("String matched at postion %d",k);
	}
}



*/



/*
#include <stdio.h>
#include <string.h>
void rabinKarp(char sub[], char Main[], int q) {
  	int m = strlen(sub);
  	int n = strlen(Main);
  	int count=0;
  	int i, j;
  	int s = 0;
  	int M = 0;
  	int h = 1;
  	for (i = 0; i < m - 1; i++)
    	h = (h * 10) % q;
  	for (i = 0; i < m; i++) {
    	s = (10 * s + sub[i]) % q;
    	M = (10 * M + Main[i]) % q;
  	}
  	for (i = 0; i <= n - m; i++) {
    	if (s == M) {
    		count++;
      		for (j = 0; j < m; j++) {
        		if (Main[i + j] != sub[j])
          		break;
      			}
			if (j == m){
				printf("Pattern is found at position: %d \n", i);
			}
    	}
    	if (i < n - m) {
      		M = (10 * (M - Main[i] * h) + Main[i + m]) % q;

      	if (M < 0)
        	M = (M + q);
    	}
  	}
  	printf("No. of spurious hits : %d",count);
}

int main() {
  	char Main[]="2359023141526739921";
  	char sub[]="31415";
  	int q = 10;
  	rabinKarp(sub, Main, q);
}*/


/*void lcsAlgo(char S1[], char S2[], int m, int n) {
  int LCS_table[m + 1][n + 1];
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      if (i == 0 || j == 0)
        LCS_table[i][j] = 0;
      else if (S1[i - 1] == S2[j - 1])
        LCS_table[i][j] = LCS_table[i - 1][j - 1] + 1;
      else
        LCS_table[i][j] = max(LCS_table[i - 1][j], LCS_table[i][j - 1]);
    }
  }

  int index = LCS_table[m][n];
  char lcsAlgo[index + 1];
  lcsAlgo[index]= ' ';
  int i = m, j = n;
  while (i > 0 && j > 0) {
    if (S1[i - 1] == S2[j - 1]) {
      lcsAlgo[index - 1] = S1[i - 1];
      i--;
      j--;
      index--;
    }

    else if (LCS_table[i - 1][j] > LCS_table[i][j - 1])
      i--;
    else
      j--;
  }
  cout << "S1 : " << S1 << "\nS2 : " << S2 << "\nLCS: " << lcsAlgo << "\n";
}

void lcsAlgo(char S1[], char S2[], int m,int n){
	int LCS_table[m+1][n+1];
	for(int i=0;i<=m;i++){
		for(int j=0;j<=n;j++){
			if(i==0 || j==0)
				LCS_table[i][j]==0;
			
			else if(S1[i-1]==S2[j-1])
				LCS_table[i][j]=LCS_table[i-1][j-1]+1;
			
			else
				LCS_table[i][j]=max(LCS_table[i-1][j],LCS_table[i][j-1]);
			
		}
	}
	for(int i=0;i<=m;i++){
		for(int j=0;j<=n;j++){
			cout<<LCS_table[i][j]<<" ";
		}
		cout<<"\n";
	}
	cout<<m<<"\n";
	cout<<n<<"\n";
	int index=LCS_table[m][n];
	cout<<index;
	char lcsA[index+1];
	lcsA[index]='\0';
	int i=m;
	int j=n;
	cout<<"here";
	while(i>0 && j>0){
		if(S1[i-1]==S2[j-1]){
			lcsA[index-1]=S1[i-1];
			i--;
			j--;
			index--;
		}
		else if(LCS_table[i-1][j]>LCS_table[i][j-1]){
			i--;
		}
		else{
			j--;
		}
	}
	cout<<"here";
	cout<<lcsA;
}

int main() {
  char S1[] = "ACADB";
  char S2[] = "CBDA";
  int m = strlen(S1);
  int n = strlen(S2);
  lcsAlgo(S1, S2, m, n);
}

#include <stdio.h>
int is_attack(int i, int j, int board[6][6],int N) {
  int k, l;
  for(k=1; k<=i-1; k++) {
    if(board[k][j] == 1)
      return 1;
  }
  k = i-1;
  l = j+1;
  while (k>=1 && l<=N) {
    if (board[k][l] == 1)
      return 1;
    k=k+1;
    l=l+1;
  }
  k = i-1;
  l = j-1;
  while (k>=1 && l>=1) {
    if (board[k][l] == 1)
      return 1;
    k=k-1;
    l=l-1;
  }
  return 0;
}

int n_queen(int r, int n, int N, int board[6][6]){
  if (n==0)
    return 1;
  int j;
  for (j=1; j<=N; j++) {
    if(!is_attack(r, j, board, N)) {
      board[r][j] = 1;

      if (n_queen(r+1, n-1, N, board))
        return 1;
      board[r][j] = 0;
    }
  }
  return 0;
}
int main() {
  int board[6][6];
  int i, j;
  for(i=0;i<=5;i++) {
    for(j=0;j<=5;j++)
      board[i][j] = 0;
  }
  n_queen(1, 5, 5, board);
  for(i=1;i<=5;i++) {
    for(j=1;j<=5;j++){
    	if(board[i][j]==0){
    		printf(" . ");
		}
		if(board[i][j]==1){
			printf(" Q ");
		}
	}
    printf("\n");
  }
  return 0;
}
*/



/*
class Edge {
	public:
    int source;
    int destination;
    int weight;
};

class Graph {
	public:
    int V;  
    int E;
    struct Edge edges[1000];
};

void bellman_ford( Graph* graph, int source) {
    int distances[1000];
    int i, j;
    
    for (i = 0; i < graph->V; ++i) {
        distances[i] = INT_MAX;
    }
    distances[source] = 0;
    for (i = 0; i < graph->V-1; ++i) {
        for (j = 0; j < graph->E; ++j) {
            int u = graph->edges[j].source;
            int v = graph->edges[j].destination;
            int weight = graph->edges[j].weight;
            if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
            }
        }
    }
    for (j = 0; j < graph->E; ++j) {
        int u = graph->edges[j].source;
        int v = graph->edges[j].destination;
        int weight = graph->edges[j].weight;
        if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
            printf("Graph contains a negative-weight cycle\n");
            return;
        }
    }
    printf("Vertex Distance from source\n");
    for (i = 0; i < graph->V; ++i) {
        printf("%d \t\t %d\n", i, distances[i]);
    }
}

int main() {
    int V, E, i, source;
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    
    printf("Enter the number of vertices: ");
    scanf("%d", &V);
    
    printf("Enter the number of edges: ");
    scanf("%d", &E);
    
    graph->V = V;
    graph->E = E;
    
    for (i = 0; i < E; ++i) {
        printf("Enter source, destination and weight of edge %d: ", i+1);
        scanf("%d %d %d", &graph->edges[i].source, &graph->edges[i].destination, &graph->edges[i].weight);
    }
    
    printf("Enter the source node: ");
    scanf("%d", &source);
    
    bellman_ford(graph, source);
    
    return 0;
}




class Edge{
	public:
	int source;
	int destination;
	int weight;
};
class Graph{
	public:
	int V;
	int E;
	Edge edges[1000];
};
void BellmanFord(Graph* graph, int source){
	int distances[1000];
	for(int i=0;i<graph->V;i++){
		distances[i]=1000;
	}
	distances[source]=0;
	for(int i=0;i<graph->V-1;i++){
		for(int j=0;j<graph->E;j++){
			int u=graph->edges[j].source;
			int v=graph->edges[j].destination;
			int w=graph->edges[j].weight;
			if(distances[u]!=1000 && distances[u]+w<distances[v]){
				distances[v]=distances[u]+w;
			}
		}
	}
	for(int j=0;j<graph->E;j++){
		int u=graph->edges[j].source;
		int v=graph->edges[j].destination;
		int w=graph->edges[j].weight;
		if(distances[u]!=1000 && distances[u]+w<distances[v]){
			cout<<"infinite self loop";
			return;
		}
	}
	for(int i=0;i<graph->V;i++){
		cout<<i<<","<<distances[i];
	}
}
int main(){
	int source;
	Graph* graph=(Graph*)malloc(sizeof(Graph));
	int V;
	int E;
	cin>>V;
	cin>>E;
	for(int i=0;i<E;i++){
		printf("Enter source, destination and weight of edge %d: ", i+1);
        scanf("%d %d %d", &graph->edges[i].source, &graph->edges[i].destination, &graph->edges[i].weight);
	}
	BellmanFord(graph,source);
}


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<string.h>
using namespace std;
void computeLPSArray(char* pat, int M, int* lps)
{
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else 
            if (len != 0) {
                len = lps[len - 1];
 
            }
            else 
            {
                lps[i] = 0;
                i++;
            }
        }
}
void KMPSearch(char* pat, char* txt)
{
    int m= strlen(pat);
    int n = strlen(txt);
    int lps[m];
    computeLPSArray(pat, m, lps);
 
    int i = 0; 
    int j = 0;
    while ((n - i) >= (m - j)) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }
        if (j == m) {
            printf("Found pattern at index %d ", i - j);
            j = lps[j - 1];
        }
        else if (i < n && pat[j] != txt[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }
}
int main()
{
    char txt[] = "010203023343534";
    char pat[] = "334";
    KMPSearch(pat, txt);
    
}


#include <bits/stdc++.h>
using namespace std;

struct Point {int x, y;};
struct Segment {Point p, q;};

struct Event {int x, y, i; bool t;};
bool operator<(const Event& e1, const Event& e2) {return e1.y == e2.y ? e1.x < e2.x : e1.y < e2.y;}
bool isLeft(Point a, Point b, Point c) {return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y) > 0;}
bool onSegment(Point p, Point q, Point r) {return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);}
bool intersect(Segment s1, Segment s2) {return isLeft(s1.p, s1.q, s2.p) != isLeft(s1.p, s1.q, s2.q) && isLeft(s2.p, s2.q, s1.p) != isLeft(s2.p, s2.q, s1.q) || isLeft(s1.p, s1.q, s2.p) == 0 && onSegment(s1.p, s2.p, s1.q) || isLeft(s1.p, s1.q, s2.q) == 0 && onSegment(s1.p, s2.q, s1.q) || isLeft(s2.p, s2.q, s1.p) == 0 && onSegment(s2.p, s1.p, s2.q) || isLeft(s2.p, s2.q, s1.q) == 0 && onSegment(s2.p, s1.q, s2.q);}

int countIntersections(vector<Segment>& segs) {
    vector<Event> evs;
    for (int i = 0; i < segs.size(); ++i) {
        evs.push_back({segs[i].p.x, segs[i].p.y, i, true});
        evs.push_back({segs[i].q.x, segs[i].q.y, i, false});
    }
    sort(evs.begin(), evs.end());

    int ans = 0;
    set<int> active;
    for (const auto& e : evs) {
        int i = e.i;
        if (e.t) {
            for (const auto& j : active) {
                if (intersect(segs[i], segs[j])) {
                    ++ans;
                }
            }
            active.insert(i);
        } else {
            active.erase(i);
        }
    }
    return ans;
}

int main() {
    vector<Segment> segs = {{{1, 5}, {4, 5}}, {{2, 5}, {10, 1}},{{3, 2}, {10, 3}},{{6, 4}, {9, 4}},{{7, 1}, {8, 1}}};
    cout << "Intersections: " << countIntersections(segs) << '\n';
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<string.h>
#include <vector>
using namespace std;
struct Mypoint{
    int x, y;
};
int orientation(Mypoint p, Mypoint q, Mypoint r){
    int val = (q.y - p.y) * (r.x - q.x) -
            (q.x - p.x) * (r.y - q.y);


    if (val == 0) return 0; 
    return (val > 0)? 1: 2; 
}
void convexHull(Mypoint pointval[], int n)
{
    if (n < 3) return;
    vector<Mypoint> hull;
    int l = 0;
    for (int i = 1; i < n; i++)
        if (pointval[i].x < pointval[l].x)
            l = i;
    int p = l, q;
    do
    {
        hull.push_back(pointval[p]);
        q = (p+1)%n;
        for (int i = 0; i < n; i++)
        {
        if (orientation(pointval[p], pointval[i], pointval[q]) == 2)
            q = i;
        }   
        p = q;


    } while (p != l); 
    for (int i = 0; i < hull.size(); i++)
        cout << "(" << hull[i].x << ", "
            << hull[i].y << ")\n";
}
int main()
{
    Mypoint pointval[] = {{1, 3}, {1, 2}, {4, 1}, {2, 5},
                    {3, 1}, {2, 0}, {3, 2}};
    int n = sizeof(pointval)/sizeof(pointval[0]);
    convexHull(pointval, n);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
struct Edge {
    int source;
    int destination;
    int weight;
};

struct Graph {
    int V;  
    int E;
    struct Edge edges[1000];
};

void bellman_ford(struct Graph* graph, int source) {
    int distances[1000];
    int i, j;
    
    for (i = 0; i < graph->V; ++i) {
        distances[i] = 1000;
    }
    distances[source] = 0;
    for (i = 0; i < graph->V-1; ++i) {
        for (j = 0; j < graph->E; ++j) {
            int u = graph->edges[j].source;
            int v = graph->edges[j].destination;
            int weight = graph->edges[j].weight;
            if (distances[u] != 1000 && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
            }
        }
    }
    for (j = 0; j < graph->E; ++j) {
        int u = graph->edges[j].source;
        int v = graph->edges[j].destination;
        int weight = graph->edges[j].weight;
        if (distances[u] != 1000 && distances[u] + weight < distances[v]) {
            printf("Graph contains a negative-weight cycle\n");
            return;
        }
    }
    printf("Vertex   Distance from source\n");
    for (i = 0; i < graph->V; ++i) {
        printf("%d \t\t %d\n", i, distances[i]);
    }
}

int main() {
    int V, E, i, source;
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    
    printf("Enter the number of vertices: ");
    scanf("%d", &V);
    
    printf("Enter the number of edges: ");
    scanf("%d", &E);
    
    graph->V = V;
    graph->E = E;
    
    for (i = 0; i < E; ++i) {
        printf("Enter source, destination and weight of edge %d: ", i+1);
        scanf("%d %d %d", &graph->edges[i].source, &graph->edges[i].destination, &graph->edges[i].weight);
    }
    
    printf("Enter the source node: ");
    scanf("%d", &source);
    
    bellman_ford(graph, source);
    
    return 0;
}

#include <stdio.h>
#include <string.h>
void rabinKarp(char sub[], char Main[], int q) {
  	int m = strlen(sub);
  	int n = strlen(Main);
  	int count=0;
  	int i, j;
  	int s = 0;
  	int M = 0;
  	int h = 1;
  	for (i = 0; i < m - 1; i++)
    	h = (h * 10) % q;
  	for (i = 0; i < m; i++) {
    	s = (10 * s + sub[i]) % q;
    	M = (10 * M + Main[i]) % q;
  	}
  	for (i = 0; i <= n - m; i++) {
    	if (s == M) {
    		count++;
      		for (j = 0; j < m; j++) {
        		if (Main[i + j] != sub[j]){
        			
        			break;
				}
          		
      			}
			if (j == m){
				count--;
				printf("Pattern is found at position: %d \n", i);
			}
    	}
    	if (i < n - m) {
      		M = (10 * (M - Main[i] * h) + Main[i + m]) % q;

      	if (M < 0)
        	M = (M + q);
    	}
  	}
  	printf("No. of spurious hits : %d",count);
}

int main() {
  	char Main[]="abcdef";
  	char sub[]="bc";
  	int q = 10;
  	rabinKarp(sub, Main, q);
}
*/
#include <stdio.h>

int n;
int e;
int size[1000][1000];
int f[1000][1000];
int color[1000];
int pred[1000];

int min(int x, int y) {
  return x < y ? x : y;
}

int head, tail;
int q[1000 + 2];

void enqueue(int x) {
  q[tail] = x;
  tail++;
  color[x] = 1;
}

int dequeue() {
  int x = q[head];
  head++;
  color[x] = 2;
  return x;
}
int bfs(int start, int target) {
  int i, j;
  for (i = 0; i < n; i++) {
    color[i] = 0;
  }
  head = tail = 0;
  enqueue(start);
  pred[start] = -1;
  while (head != tail) {
    i = dequeue();
    for (j = 0; j < n; j++) {
      if (color[j] == 0 && size[i][j] - f[i][j] > 0) {
        enqueue(j);
        pred[j] = i;
      }
    }
  }
  return color[target] == 2;
}
int Fulkerson(int source, int sink) {
  int i, j, k;
  int mf= 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      f[i][j] = 0;
    }
  }
  while (bfs(source, sink)) {
    int increment = 100000000;
    for (k = n - 1; pred[k] >= 0; k = pred[k]) {
      increment = min(increment, size[pred[k]][k] - f[pred[k]][k]);
    }
    for (k = n - 1; pred[k] >= 0; k = pred[k]) {
      f[pred[k]][k] += increment;
      f[k][pred[k]] -= increment;
    }
    mf+= increment;
  }
  return mf;
}

int main() {
	printf("21BCE3503- Naethen Luke\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      size[i][j] = 0;
    }
  }
  n = 6;
  e = 7;

  size[0][1] = 3;
  size[0][4] = 8;
  size[1][2] = 2;
  size[2][4] = 1;
  size[2][5] = 4;
  size[3][5] = 5;
  size[4][2] = 9;
  size[4][3] = 6;

  int s = 0, t = 5;
  printf("Max Flow: %d\n",Fulkerson(s, t));
}




