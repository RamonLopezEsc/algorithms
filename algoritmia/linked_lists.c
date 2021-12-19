#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *next;
} node_t;

void push_end(node_t **head, int value)
{
    node_t *current = *head;
    if (*head == NULL)
    {
        current = malloc(sizeof(node_t));
        (*current).data = value;
        (*current).next = NULL;
        *head = current;
    }
    else
    {
        while ((*current).next != NULL)
        {
            current = (*current).next;
        }
        (*current).next = malloc(sizeof(node_t));
        (*(*current).next).data = value;
        (*(*current).next).next = NULL;
    }
}

void push_init(node_t **head, int value)
{
    node_t *new_node;
    new_node = malloc(sizeof(node_t));
    (*new_node).data = value;
    (*new_node).next = *head;
    *head = new_node;
}

void del_first(node_t **head)
{
    if (*head == NULL)
    {
        printf("Underflow!");
        exit(0);
    }
    else
    {
        node_t *aux;
        aux = (*(*head)).next;
        free(*head);
        *head = aux;
    }
}

void del_last(node_t **head)
{
    node_t *current = *head;
    if (*head == NULL)
    {
        printf("Underflow!");
        exit(0);
    }
    else
    {
        while ((*current).next != NULL)
        {
            current = (*current).next;
            printf("%i", (*current).data);
        }
    }
}

void print_list(node_t *head)
{
    node_t *current = head;
    while (current != NULL)
    {
        printf("%d\n", (*current).data);
        current = (*current).next;
    }
}

int main()
{
    node_t *head;
    head = malloc(sizeof(node_t));
    head = NULL;

    push_end(&head, 1);
    del_first(&head);
    print_list(head);
}