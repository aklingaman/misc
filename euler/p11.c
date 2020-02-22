#include <stdio.h>
#include <stdlib.h>
#include <math.h>

long largestProduct(int nums[]);

int main() {
    FILE *open = fopen("p11num.txt","r");
    int matrix[20][20];
    for(int i = 0; i<20; i++) {
        for(int j = 0; j<20; j++) {
            fscanf(open, "%d", &matrix[i][j]); 
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    
    //Here we are assuming the answer wont be diagonal, and finding the largest product. 
    long largestRowProduct = 0;
    long largestColProduct = 0;
    
    for(int i = 0; i<20; i++) { //for each row
        
        //Compute max row sum
        for(int j = 0; j<17; j++) {
            if((j!=16)&&(matrix[i][j+4]>matrix[i][j])) {
                continue;
            } else {
                long product = matrix[i][j]*matrix[i][j+1]*matrix[i][j+2]*matrix[i][j+3]; //Consideration: checking for zeroes may be faster, if needed
                if(product>largestRowProduct) {
                    largestRowProduct = product;
                }
            }
        }
        //Compute max row sum
        
        //Compute max column sum
        for(int j = 0; j<17; j++) { //Just switch i and j! 
            if((j!=16)&&(matrix[j+4][i]>matrix[j][i])) { //Yay for shortcircuiting. (otherwise we would be checking the first element of the next row, which is bad)
                continue;
            } else {
                long product = matrix[j][i]*matrix[j+1][i]*matrix[j+2][i]*matrix[j+3][i]; //Consideration: checking for zeroes may be faster, if needed
                if(product>largestColProduct) {
                    largestColProduct = product;
                }
            }
        }
    }
    printf("Largest row product: %ld\n", largestRowProduct);
    printf("Largest col product: %ld\n", largestColProduct);

    //So that didnt work, the answer must be a diagonal one. Lovely.
    long largestDRProduct = 0; //DR = down right.
    long largestDLProduct = 0; //DL = down left. 
    for(int i = 0; i<20; i++) { //16 is the last beginning of a run going down and right. 3 is the first element that can go down left. 

    //Down right
    //Here we can ignore all cases that have i > 16
    if(i>16) { continue; }
    //Same principle as before, if the element we will be adding next iteration is larger than the removal one, dont compute.
    for(int j = 0; j<17; j++) {
        
        if((j!=17)&&(matrix[i+4][j+4]>matrix[i][j])) {
            continue;
        } else {
            long product = matrix[i][j]*matrix[i+1][j+1]*matrix[i+2][j+2]*matrix[i+3][j+3]; //Consideration: checking for zeroes may be faster, if needed
            if(product>largestDRProduct) {
                largestDRProduct = product;
            }
        }   
    }
    //Down right

    //Down left
    if(i<3) { continue; }
    for(int j = 3; j<20; j++) {
        if((j!=3)&&(matrix[i+4][j-4]>matrix[i][j])) {
            continue;
        } else {
            long product = matrix[i][j]*matrix[i+1][j-1]*matrix[i+2][j-2]*matrix[i+3][j-3]; //Consideration: checking for zeroes may be faster, if needed
            if(product>largestDLProduct) {
                largestDLProduct = product;
            }
        }   

 
    //Down left
    }
}
    printf("Largest DR product: %ld\n", largestDRProduct);
    printf("Largest DL product: %ld\n", largestDLProduct);

    fclose(open);
}

//Finds largest product of 4 consecutive entries. 
long largestProduct(int nums[]) {
    long largestProduct = 0;
    for(int i = 0; i<17; i++ ) { //16 is the last starter element of a run. 
       // printf("LP: %d\n", nums[i]);
        if(nums[i+4]>=nums[i]) {//If the number that would be included is larger than the one were removing.... keep going but dont calculate
            continue;
        } else {
            int product = nums[i]*nums[i+1]*nums[i+2]*nums[i+3]; //i+4th element is the one we dont want to include ( at the cost of the ith )
         //   printf("LP compute: %d", product);
            if(product>largestProduct) {
                largestProduct = product;
            }
        }
    }
    return largestProduct;
}
