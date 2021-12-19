#include <stdio.h>

int main()
{
    int i, j, temp, n;
    int list[10] = {23, 17, 5, 90, 12, 44, 38, 84, 77, 1};

    n = sizeof(list)/sizeof(int);

    for (i = 0; i < n - 1; i++)
    {
        for (j = 0; j < n - 1; j++)
        {
            if (list[j] < list[j+1])
            {
                /* Do nothing */
                ;
            }
            else
            {
                temp = list[j+1];
                list[j+1] = list[j];
                list[j] = temp;
            }
        }
    }

    printf("La lista ordenada queda:\n");
    for (i = 0; i < n; i++)
    {
        printf("%i ", list[i]);
    }

    return 0;
}