# SGEMM

for i = 1:6 
    N = 512*(2^i);
    A = single(rand(N,N));
    B = single(rand(N,N));

    start = clock();
    C = A * B;
    elapsedTime = etime(clock(), start);

    gFlops = 2*N*N*N/(elapsedTime * 1e+9);
    printf("Elapsed Time [%d]: %.3f ms, %.3f GFlops\n", N, elapsedTime, gFlops);
end