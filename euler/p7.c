#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct node_ {
    int data;
    struct node_ *next;
}Node;

int insert(Node *head, int num);
void clear(Node *head);
int isPrime(Node *head, int num);

int main() {
    int size = 1;
    int num = 3;
    Node *head = malloc(sizeof(Node));
    head->data = 2;
    while(size<10001) {
        printf("num: %d, size: %d\n",num,size);
        if(isPrime(head,num)) {
            insert(head,num);
            size++;
        }
        num++;
    }
    printf("%d\n",num-1);
    clear(head);
}

//Makes new Node at the end of list containing num. Will iterate through checking, if it finds a duplicate, will cancel, returning 0, else retruning 1. Assumes a non empty list. Runs in O(n) because im lazy.
int insert(Node *head, int num) {
    if(head->next==NULL) {
        Node *new = malloc(sizeof(Node));
        new->data = num;
        head->next = new;
    } else {
        insert(head->next,num);
    }
}

//Clears all the data in the list that was malloc'd. 
void clear(Node *head) {
    if(head->next!=NULL) {
        clear(head->next);
        free(head);
    }
}

//Iterates through the list trying to divide the list contents into the number for a clean division. If it manages to get through, return 1, else 0.
int isPrime(Node *head, int num) {
    //printf("\tisPrime? num: %d\n",num);
    if(num%(head->data)) {
        //printf("\t\t checking %d\n", head->data);
        if(head->next!=NULL) {
            return isPrime(head->next, num);
        } else {
            return 1;
        }
    } else {
        //printf("\t\t(%d)%%(%d)!=0\n",num,head->data);
        return 0;
    }
}



