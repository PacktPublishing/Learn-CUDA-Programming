N = 8192;
A = single(rand(N,N));
B = single(rand(N,N));

start = clock();
C = A * B; 
elapsedTime = etime(clock(), start);

gFlops = 2*N*N*N/(elapsedTime * 1e+9);
fprintf("Elapsed Time: %.3f ms, %.3f GFlops\n", elapsedTime, gFlops);