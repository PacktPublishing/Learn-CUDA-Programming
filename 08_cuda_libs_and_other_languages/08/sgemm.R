# Matrix Multiplication using R
for(i in seq(1:6)) {
    N = 512*(2^i)
    A = matrix(rnorm(N^2, mean=0, sd=1), nrow=N) 
    B = matrix(rnorm(N^2, mean=0, sd=1), nrow=N) 
    elapsedTime = system.time({C = A %*% B})[3]
    gFlops = 2*N*N*N/(elapsedTime * 1e+9);
    print(sprintf("Elapsed Time [%d]: %3.3f ms, %.3f GFlops", N, elapsedTime, gFlops))
}