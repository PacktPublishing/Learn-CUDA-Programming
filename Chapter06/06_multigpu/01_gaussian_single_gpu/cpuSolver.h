
int gaussianEliminationCPU(unsigned int* AB, int rowCount, int columnCount)
{
    int row, pivot_row, col, reduce_row;
    int *isRowReduced;
    int packedColumnCount;
    int matrixElement;
    int packIndex, bitIndex;

    packedColumnCount = intCeilDiv(columnCount + 1, PACK_SIZE);

    isRowReduced = (int*) malloc(rowCount * sizeof(int));

    if(isRowReduced == NULL)
    {
        printf("Failed to allocate space for CPU Gauss solver. Quitting.\n");
        return -1;
    }

    memset(isRowReduced, 0, rowCount * sizeof(int));

    for(row = 0; row < columnCount; row++)
    {
        
        // Find pivot
        for(pivot_row = 0; pivot_row < rowCount; pivot_row++)
        {
            packIndex = pivot_row * packedColumnCount + row / PACK_SIZE;
            bitIndex  = row % PACK_SIZE;

            matrixElement = (AB[packIndex] >> bitIndex) & 1;
            if(isRowReduced[pivot_row] == 0 && matrixElement != 0)
            {
                //pivote found
                isRowReduced[pivot_row] = 1;
                break;
            }
        }

        if(pivot_row == rowCount)
        {
           printf("No suitable pivot was found for (%d, %d) position\n", row, row);
           return row;
        }

        // Use row 'row' to reduce equations row+1 to number_of_equations-1
        int bitIndex = row % PACK_SIZE;
        int r = row / PACK_SIZE;
        int pivotIndex = pivot_row*packedColumnCount;
        for(reduce_row = 0; reduce_row < rowCount; reduce_row++)
        {
            if(isRowReduced[reduce_row] == 1) 
                continue;

            packIndex = reduce_row * packedColumnCount + r;

            matrixElement = (AB[packIndex] >> bitIndex) & 1;

            if(matrixElement == 0)
            {
                continue;
            }

            // actual reduction
            for(col = r; col < packedColumnCount; col++)
            {
                //loopCount++;
                AB[reduce_row*packedColumnCount + col] ^= AB[pivotIndex + col];
            }
        }
    }

  //  printf ("LoopCount is %d\n", loopCount);
    free(isRowReduced);
    return 0;
}

int backsubstitution(unsigned char* matrixA, unsigned char* B, unsigned char* x, int number_of_equations, int number_of_variables)
{
    int row, col, var;
    int* pivoteRowIndex;
    long int index;

    memset(x, 0, sizeof(unsigned char) * number_of_variables);

    pivoteRowIndex = (int*) malloc(sizeof(int)*number_of_equations);

    for(row = 0; row < number_of_equations; row++)
    {
        pivoteRowIndex[row] = -1;
    }
    
    for(row = 0; row < number_of_equations; row++)
    {
        for(col = 0; col < number_of_variables; col++)
        {
            index = (long int)row*number_of_variables+col;
            if(matrixA[index] != 0)
                break;
        }
        if(col != number_of_variables)
        {
            //printf("Row %d is pivot for col %d\n", row, col);
            pivoteRowIndex[col] = row;
        }
        else
        {
            // printf("Row %d is NULL\n", row);
        }
    }

    for(var = number_of_variables - 1; var >= 0; var--)
    {
        if(pivoteRowIndex[var] == -1)
        {
            printf("backsubstitution failed at %d\n", var);
            break;
        }
        row = pivoteRowIndex[var];
        x[var] = B[row];
        for(col = var+1; col < number_of_variables; col++)
        {
            index = (long int)row*number_of_variables+col;
            x[var] ^= (matrixA[index] & x[col]);
        }
    }

    free(pivoteRowIndex);
    return 0;
}


void gaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount)
{
    
    int packedColumnCount;
    unsigned int* packedAB;
    
    packedColumnCount = intCeilDiv(columnCount + 1, PACK_SIZE);
 
    // --- Allocate memory for input matrix A on cpu
    long int packedSize = ((long int)sizeof(unsigned int)) * rowCount * packedColumnCount;
    packedAB           = (unsigned int*) malloc(packedSize);
    if(packedAB == NULL)
    {
        printf("Unable to allocate space for packed linear system.\n");
        return;
    }

    packLinearSystem(packedAB, A, B, rowCount, columnCount);
    gaussianEliminationCPU(packedAB, rowCount, columnCount);
    unpackLinearSystem(A, B, packedAB, rowCount, columnCount);

    return;
}
