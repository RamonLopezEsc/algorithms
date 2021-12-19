#include <stdio.h>
#include <math.h>

void suma_elem_array(int num_data, int *array_x, int *array_y, float *sum_x, float *sum_y, float *sum_xy,
                     float *sum_cuad_x, float *sum_cuad_y)
{
    /* ---------------------------------- */
    int i;
    /* ---------------------------------- */

    for (i = 0; i < num_data; i++)
    {
        *sum_x += *(array_x + i);
        *sum_y += *(array_y + i);
        *sum_xy += *(array_x + i) * *(array_y + i);
        *sum_cuad_x += *(array_x + i) * *(array_x + i);
        *sum_cuad_y += *(array_y + i) * *(array_y + i);
    }
}

void calc_parametros(float *a, float *b, double *r, int num_data, float sum_x, float sum_y, float sum_xy,
                     float sum_cuad_x, float sum_cuad_y)
{
    *a = ((sum_y * sum_cuad_x) - (sum_x * sum_xy)) / ((num_data * sum_cuad_x) - (sum_x * sum_x));
    *b = ((num_data * sum_xy) - (sum_x * sum_y)) / ((num_data * sum_cuad_x) - (sum_x * sum_x));
    *r = ((num_data * sum_xy) - (sum_x * sum_y)) / (sqrt((((num_data * sum_cuad_x) - (sum_x * sum_x))
                                                    * ((num_data * sum_cuad_y) - (sum_y * sum_y)))));
}

void main()
{
    /* ---------------------------------- */
    double r = 0;
    float a = 0.0, b = 0.0;
    float sum_x = 0,sum_y = 0, sum_xy = 0, sum_cuad_x = 0, sum_cuad_y = 0;
    int arr_x[] = {15, 19, 25, 23, 34, 40};
    int arr_y[] = {80, 70, 60, 40, 20, 10};
    int num_data = sizeof(arr_x) / sizeof(arr_x[0]);
    /* ---------------------------------- */

    suma_elem_array(num_data, &arr_x[0], &arr_y[0], &sum_x, &sum_y, &sum_xy, &sum_cuad_x, &sum_cuad_y);
    calc_parametros(&a, &b, &r, num_data, sum_x, sum_y, sum_xy, sum_cuad_x, sum_cuad_y);

    printf("ECUACION DE LA RECTA\n");
    printf("%f + (%f)x\n\n", a, b);
    printf("COEFICIENTE DE PEARSON\n");
    printf("%lf", r);
}