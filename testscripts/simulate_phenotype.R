# writen by talia
simulate_phenotype <- function(num_cau_SNP=20, num_SNP=500, samplesize=20, h_squared=0.5){
  # generate genotype in Binomial distribution
  pj <- runif(num_SNP, 0.01, 0.5) # probability of success on each trial
  xij_star <- matrix(0, samplesize, num_SNP)
  #for every SNP, select from 0, 1, 2
  for (j in 1: num_SNP) {
    xij_star[,j] <- rbinom(samplesize, 2, pj[j]) } # 2, number of trials
  
  #position of causal SNPs
  CauSNP <- sample(1:num_SNP, num_cau_SNP, replace = F)
  Ord_CauSNP <- sort(CauSNP, decreasing = F)
  
  # generate beta, which is the best predictor
  beta <- rep(0,num_SNP)
  dim(beta) <- c(num_SNP,1)
  # non-null betas follow standard normal distribution
  beta[Ord_CauSNP] <- rnorm(num_cau_SNP,0,1)
  
  # epsilon
  var_e <- sum((xij_star %*% beta)^2) # Multiplies two matrices
  # var_e <- t(beta)%*%t(xij_star)%*%xij_star%*%beta/samplesize*(1-h_squared)/h_squared
  e <- rnorm(samplesize, 0,sqrt(var_e))
  dim(e) <- c(samplesize, 1)
  
  # generate phenotype
  pheno <- xij_star %*% beta + e
  # scale(genotype matrix)
  return(pheno)
}