function bets = RSPNBC(A,betas,normalized,exclude_st,C,is_directed,print_prog)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION bets = RSPNBC(A,betas,normalized,exclude_st,C)
% 
% The randomized shortest paths net betweenness centrality from 
% Kivim\"aki et al.: Two betweenness centralities based on Randomized
% Shortest Paths (2015).
% 
% Computes the RSP net betweenness as the sum of expected net flows along
% the edges connected to a node.
%
% INPUT ARGUMENTS:
% 
% A:           nxn weighted non-negative adjacency matrix, of stronly
%              connected graph G.
%
% betas:       The vector of inverse temperature parameters. With one
%              beta the function returns a vector of betweenness scores, 
%              with several betas, the function returns a matrix whose
%              columns are betweenness vectors.
%
% normalized:  If 1, the results are normalized with the number of
%              node-pairs (default = 0).
% 
% exclude_st: If 1, s and t get 0 betweenness over all s-t-walks
%             (default = 0). 
%
% C:           nxn matrix, cost matrix C associated to G (default = 1./A)
% 
% is_directed: If 1, G is considered directed (this is checked by the 
%              algorithm by default)
% 
% print_prog:  If 1, prints a fancy progressbar (default=1, if n>1000) 
%
% OUTPUT ARGUMENTS:
% bets:        The RSP net betweenness vector, or matrix of betweenness
%              vectors.
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
% bet = RSPNBC(A,beta)
% 
% (c) 2014 I. Kivim\"aki
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
[n1,n2] = size(A);
if n1 ~= n2
    error('Martix A must be square')
else
    n = n1;
end

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

if nargin < 6
  is_directed = any(any(A ~= A'));
end

if nargin < 7
  print_prog = n>1000;
end

e = ones(n,1);
I = eye(size(A)); % Identity matrix

degs = A*e;
if issparse(A)
  D_1 = spdiags(1./degs,0,n,n); % The generalized outdegree matrix
else
  D_1 = diag(1./degs);
end

P_ref = D_1*A; % The reference transition probability matrix

nbetas = numel(betas);
bets = zeros(n,nbetas);

if is_directed
  A_links = A;
else
  A_links = tril(A)>0;
end

for beta_ind = 1:nbetas
  beta = betas(beta_ind);
  if nbetas > 1
    fprintf('Computing for beta = %d (%d/%d)\n', beta, beta_ind, nbetas);
  end

  W = P_ref.*exp(-beta*C);
  
  % Check that W is valid:
  rsums = sum(W,2);
  if sum(rsums==0) > 0
    error('Beta is too large or graph has isolated nodes (some rows of W sum to zero)')
  end

  Z = (I-W)\I; % The fundamental matrix
  
  if any(any(Z<=0 | isinf(Z)))
    error('Z contains zero or Inf values - either graph is not strongly connected or beta is too small/large');
  end
  
  bet = zeros(n,1);
  
  for i = 1:n
    if print_prog
      progressbar(i,n);
    end
    z_ci = Z(:,i);
    z_ri = Z(i,:);
    
    js = find(A_links(i,:));
    
    for jj = 1:numel(js)
      j = js(jj);

      z_rj = Z(j,:);
      z_cj = Z(:,j);
    
      % First term of flow from i to j:
      s1_ij = z_ci*z_rj./Z;
      % First term of flow from j to i:
      s1_ji = z_cj*z_ri./Z;

      % Second term of flow from i to j:
      s2_vec_ij = z_ci.*z_rj'./diag(Z);
      s2_ij = e*s2_vec_ij';
      % Second term of flow from j to i:
      s2_vec_ji = z_cj.*z_ri'./diag(Z);
      s2_ji = e*s2_vec_ji';

      % Matrix of flows from i to j:
      N_ij = W(i,j)*(s1_ij - s2_ij);
      % Matrix of flows from j to i:
      N_ji = W(j,i)*(s1_ji - s2_ji);
      
      % Matrix of net flows over (i,j):
      N = abs(N_ij - N_ji);
      
      e_i = e;
      e_i(i) = 0;

      % Contribution to node i:
      if exclude_st
        bet(i) = bet(i) + e_i'*N*e_i;
      else
        bet(i) = bet(i) + e'*N*e;
      end
      
      % For undirected graphs the effect of edge (i,j) on i and j is the
      % same:
      if ~is_directed
        e_j = e;
        e_j(j) = 0;
        if exclude_st
          bet(j) = bet(j) + e_j'*N*e_j;
        else
          bet(j) = bet(j) + e'*N*e;
        end
      end
    end
    
  end
  
  bet = bet./2;

  if normalized
    if exclude_st
      bet = bet/((n-1)*(n-2));
    else
      bet = bet/((n-1)*(n));
    end
  end

  bets(:,beta_ind) = bet;
  if print_prog
    fprintf('\n');
  end
end



function progressbar(m,M)
if m == 1
  fprintf('[                    ]');
else
  binsize = floor(M/20);
  div = m/binsize;
  if div == floor(div)    
    for i = 1:(20-div+2)
      fprintf('\b');
    end
    fprintf('-');
    for i = 1:(20-div)
      fprintf(' ');
    end
    fprintf(']');
  end
end

