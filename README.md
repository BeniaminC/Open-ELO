# Open-ELO
A set of elo systems written in Python. Includes a balancer.

## Elo Systems
- Bradley-Terry (https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf)
- CodeForce (https://codeforces.com/blog/entry/102)
- Elo (https://en.wikipedia.org/wiki/Elo_rating_system)
- Endure Elo (https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf)
- Trueskill (https://www.microsoft.com/en-us/research/uploads/prod/2018/03/trueskill2.pdf)
- Glicko (http://glicko.net/glicko/glicko.pdf)
- Plackett-Luce (https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf)
- Thurstone-Mosteller (https://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf)
- TopCoder (https://www.topcoder.com/community/competitive-programming/how-to-compete/ratings)


## Single and Team Rating Systems

### Individual Player Rating Systems
- Codeforce
- Elo
- Elo-MMR (simple and non-simple)
- EndureElo
- Glicko
- Topcoder

### Team Rating Systems
- Bradley-Terry (with partial)
- Elo
- Elo-MMR (simple and non-simple)
- Trueskill
- Glicko
- Plackett-Luce
- Thurstone-Mosteller (with partial)

## Rating Systems Features
- Contest Weights
- Min/Max Performances
- Team Skill Aggregations
    - Sum
    - Average
    - Min
    - Max
    - Top N/Lowest N players
- Variants
    - Logistic
    - Gaussian
- Partial Team Ratings
- Player History Events
- Robust Average
- Weight Limit
- Noob Delay (reduce the weight of individual players)
- Sigma limit (increase deviations of player ratings)
- Drifts per day (brownian effect based on time)

## Balancer Features
- Team Combinations
- Constraints
    - Set of Two Balance constraint (best two players must be on opposing teams, next two best players must be on opposing teams, etc...)
    - Rating difference constraint (based on absolute value)
    - Top K teams constraint
    - Combined All constraints (intersection of all constraints)
- Statistics
    - Sum
    - Average
    - Difference
    - Absolute Value
- Best game

## Required Libraries
- Python 12.0+
- Numpy
- Scipy
- nptyping
- trueskill

## Graphs of Balanced Games (20 players, 10 players per game)

### Bradley-Terry
![alt text](images/Bradley-Terry.png)

### Elo-MMR
![alt text](images/Elo-MMR.png)

### Glicko
![alt text](images/Glicko.png)

### Plackett-Luce
![alt text](images/Plackett-Luce.png)

### Thurstone-Mosteller
![alt text](images/Thurstone-Mosteller.png)

### Trueskill
![alt text](images/Trueskill.png)
