#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct node_ {
    int data;
    struct node_ *next;
}Node;

int insert(Node *head, int num); //Runs in O(n) 
void clear(Node *head);

int main() {
    FILE *open = fopen("something.txt","r");
    char num[1500];
    fgets(num,1500,open); 
    Node *head = malloc(sizeof(Node));
    head->data = 2;
    clear(head);
    fclose(open);
}
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

