#include <stdio.h>
#include <math.h>

// Definicion de la funcion de la funcion matematica a evaluar
double funcion(float x)
{
	return (x * x) * exp(-x * x);
}

int main()
{
	/* ---------------------------------- */
	int n;
	float i, a, b;
	double fa, fx, fb, dx, area;
	/* ---------------------------------- */
	
	printf("Ingresa el valor del limite inferior\n");
	scanf("%f", &a);

	printf("Ingresa el valor del limite superior\n");
	scanf("%f", &b);
	
	printf("Ingresa el numero de particiones\n");
	scanf("%d", &n);

	fx = 0.0;
	fa = funcion(a);
	fb = funcion(b);
	dx = (b - a)/(float)n;

	for (i = a; i < b; i += dx)
	{
		fx += funcion(i);
	}
			
	area = (fx + (fa + fb) / 2) * dx;

	printf("%0.10f\n", area);

	return 0;
}