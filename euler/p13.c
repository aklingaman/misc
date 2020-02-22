#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct num_ {
    char *data;
    int size;
}Num;

void add(Num *a, Num *b, Num *sum);
void prepend(Num *a, char b);

int main() {
    FILE *open = fopen("p13nums.txt","r");
    Num nums[100];
    for(int i = 0; i<100; i++) {
        //printf("i = %d\n",i);
        nums[i].data = malloc(sizeof(char)*51);
        if(nums[i].data == NULL) {
            printf("Error mallocing\n");
        }
        nums[i].size = 51;
        //printf("\tmalloc run:, got a pointer with value %p\n", nums[i]);
        fscanf(open,"%s",nums[i].data);
        nums[i].data[50] = '\0';
        //printf("malloc stage %d, %s\n",i,nums[i].data);
    }
    
    /* Testing printing of nums
    printf("\ntesting"); 
    char *aSpot = &(nums[0].data[nums[0].size-1]);    
    printf("\n\n%s\n", aSpot); 
    */

    /*Testing add    
    Num new;
    new.size = 1;
    new.data = malloc(1);
    new.data[0] = '\0';
    add(&nums[0],&nums[1],&new);
    printf("\n%s\n", new.data);
    */ 
    
    /* Testing prepend
    printf("before: %s\n", nums[0].data);
    prepend(&nums[0],'[');
    printf("after: %s\n", nums[0].data);
    */
    
    // Run 
    Num *sum = malloc(sizeof(Num));
    if(sum==NULL) { printf("Error mallocing sum"); } 
    sum->size=51;
    sum->data = malloc(sizeof(char)*51);
    sum->data[0] ='\0';
    memcpy(sum->data,nums[0].data,51);
    for(int i = 1; i<100; i++) {
        printf("i = %d, sumDigits: %d\n", i,sum->size);
        //printf("sumBefore: %s\n", sum->data);
        //printf("adding---: %s\n", nums[i].data);
        Num *copy = malloc(sizeof(Num));
        memcpy(copy,sum,sizeof(Num));
        add(&nums[i],copy, sum); 
        //printf("sumAfter-: %s\n\n", sum->data);
        free(nums[i].data);
        free(copy->data);
        free(copy);
    }
    printf("num: %s\n", sum->data);
    fclose(open);
}

//Strategy: go through ones digit adding into the result. turn the number into an int, and catch carries for the next number. We need to make sure there is enough size for the number. Destroys parameters. 
void add(Num *a, Num *b, Num* sum) {
     
    //printf("a = %s\nb = %s\n", a->data,b->data); 
    prepend(a, '\0'); 
    prepend(b, '\0');
    char *aSpot = &(a->data[a->size-2]); //the last digit of a.    
    char *bSpot = &(b->data[b->size-2]);    
    int carry = 0;
    while(*aSpot||*bSpot) {
        int aVal = 0;
        if(*aSpot!=0) {     
            aVal = (*aSpot)-48; 
        }
        int bVal = 0;
        if(*bSpot!=0) {     
            bVal = (*bSpot)-48;
        }
        //printf("aVal: %d, bVal: %d, carry: %d\n",aVal,bVal,carry);    
    
        int subSum = aVal + bVal + carry;
        //printf("\tsubSum: %d\n",subSum);
        carry = 0; //Remove last carry
        if(subSum>9) {
            carry = 1; //Add the next carry. 
            subSum-=10;
        }
        //printf("\tdigit to add: %c\n", subSum+48);
        prepend(sum,(subSum+48)); //Back to char values 
        //printf("\tsum: %s\n", sum->data);    
        //printf("num: %s\n" ,num->data);
        if(*aSpot) { aSpot--; }
        if(*bSpot) { bSpot--; }
        if((!(*aSpot || *bSpot))&&(carry==1)){ //A&B exhausted but we still have a carry
            prepend(sum, '1');
        }
    }
    //printf("\tsum: %s\n", sum->data);
}

void prepend(Num *a, char b) { 
    char *new = malloc(a->size+1);
    sprintf(new, "%c%s", b, a->data);
    //free(a->data); 
    a->data = new;
    a->size++; 
}
