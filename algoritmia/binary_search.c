#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void main()
{
    int counter = 0;
    int size, num, aux, lim_izq, lim_der;

    printf("Ingrese el limite superior de la lista: ");
    scanf("%i", &size);
    printf("\n");

    lim_izq = 0;
    lim_der = size;
    aux = (lim_izq + lim_der) / 2;

    srand(time(NULL));
    num = rand() % (size + 1);

    while (num != aux)
    {
        if (num < aux)
        {
            printf("DEBUG --> NUM < AUX: %i < %i\n", num, aux);
            lim_der = aux;
            aux = (lim_izq + lim_der) / 2;
            counter++;
            printf("NUEVOS LIMITES --> SUPERIOR = %i, INFERIOR: %i\n\n", lim_der, lim_izq);
        }
        else if (num > aux)
        {
            printf("DEBUG --> NUM > AUX: %i > %i\n", num, aux);
            lim_izq = aux;
            aux = (lim_izq + lim_der) / 2;
            counter++;
            printf("NUEVOS LIMITES --> SUPERIOR = %i, INFERIOR: %i\n\n", lim_der, lim_izq);
        }
        else
        {
            counter++;
        }
    }
    printf("Numero buscado: %i\n", num);
    printf("Numero encontrado: %i\n", aux);
    printf("Numero de busquedas: %i\n", counter);
    printf("\n");
    system("pause");
}