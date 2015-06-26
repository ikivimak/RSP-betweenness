function bets = RSPBC(A,betas,normalized,exclude_st,C)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION bets = RSPBC(A,betas,normalized,exclude_st,C)
% 
% The simple randomized shortest paths betweenness centrality from 
% Kivim\"aki et al.: Two betweenness centralities based on Randomized
% Shortest Paths (2015).
% 
% Computes the simple RSP betweenness as the expected number of visits to 
% each node over RSP distributions based on values in betas.
%
% INPUT ARGUMENTS:
% 
% A:          nxn weighted non-negative adjacency matrix, of stronly
%             connected graph G.
%
% betas:     The vector of (scalar) inverse temperature parameters. With one
%             beta the function returns a vector of betweenness scores, with
%             several betas, the function returns a matrix whose columns
%             are betweenness vectors.
%
% normalized: If 1, the results are normalized with the number of
%             node-pairs (default = 0).
% 
% exclude_st: If 1, s and t get 0 betweenness over all s-t-walks
%             (default = 0). 
%
% C:          nxn matrix, cost matrix C associated to G (default = 1./A)
%
% OUTPUT ARGUMENTS:
% bets:       The RSP betweenness vector, or matrix of betweenness
%             vectors.
% 
% EXAMPLE:
% 
% A = [0     1     1     1     1     0     0     0     0     0     0
%      1     0     1     1     1     0     0     0     0     0     0
%      1     1     0     1     1     0     0     0     0     0     0
%      1     1     1     0     1     0     0     0     0     0     0
%      1     1     1     1     0     1     1     0     0     0     0
%      0     0     0     0     1     0     1     0     0     0     0
%      0     0     0     0     1     1     0     1     1     1     1
%      0     0     0     0     0     0     1     0     1     1     1
%      0     0     0     0     0     0     1     1     0     1     1
%      0     0     0     0     0     0     1     1     1     0     1
%      0     0     0     0     0     0     1     1     1     1     0];
% 
% beta = 0.1;
% bet = RSPBC(A,beta)
% 
% (c) 2014 B. Lebichot, I. Kivim\"aki
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
  normalized = 0;
end

if nargin < 4
  exclude_st = 0;
end

if nargin < 5
    C = 1./A;
    C(isinf(C)) = 0;
end

[n1,n2] = size(A);
if n1 ~= n2
    error('Martix A must be square')
else
    n = n1;
end

e = ones(n,1);
I = eye(size(A)); % Identity matrix

degs = A*e;

% degs(degs>0) = 1./degs(degs>0);

if any(degs==0)
  error('Graph contains unconnected nodes');
end

if issparse(A)
  D_1 = spdiags(1./degs); % The generalized outdegree matrix
else
  D_1 = diag(1./degs);
end

P_ref = D_1*A; % The reference transition probability matrix

nbetas = numel(betas);
bets = zeros(n,nbetas);

for beta_ind = 1:nbetas
  beta = betas(beta_ind);
  if nbetas > 1
    fprintf('Computing for beta = %d (%d/%d)\n', beta, beta_ind, nbetas);
  end
  
  W = P_ref.*exp(-beta*C); % RSP's W

  % Check that W is valid:
  rsums = sum(W,2);
  if sum(rsums==0) > 0
    error('Beta is too large or graph has isolated nodes (some rows of W sum to zero)')
  end


  Z = (I-W)\I; % The fundamental matrix
  
  if any(any(Z<=0 | isinf(Z)))
    error('Z contains zero or Inf values - either graph is not strongly connected or beta is too small/large');
  end

  Zdiv = 1./Z; % the matrix containing elements 1/Z(i,j)

  Zdiv(Zdiv(:)==Inf) = 0;

  DZdiv = diag(diag(Zdiv)); % diagonal matrix containing the diagonal
                            % elements 1/Z(i,i);

  if exclude_st
    % With this form, RSPBC doesn't increase for i = s and i = t:
    bet = diag( Z * transpose(Zdiv-(n-1)*DZdiv) * Z ) - n*diag(Z);
  else
    % With this form, RSPBC increase for i = s and i = t and converges to
    % the stationary distribution (up to a constant multiplying factor):
    bet = diag( Z * transpose(Zdiv-n*DZdiv) * Z );
  end
  
  if normalized
    if exclude_st
      bet = bet/((n-1)*(n-2));
    else
      bet = bet/((n-1)*(n));
    end
  end


  bets(:,beta_ind) = bet;
end

