matrix = []
n = list(map(int, input().strip().split()))
matrix = [n]
n_len = len(n)
for _ in range(n_len-1):
    n = list(map(int, input().strip().split()))
    matrix.append(n)
def bfs(matrix):
    res = 1
    row, col = len(matrix), len(matrix[0])
    visit = [[0 for _ in range(col)] for _ in range(row)]
    queue = [[0, 0]]
    visit[0][0] = 1
    while queue:
        queue_len = len(queue)
        # print(visit)
        for _ in range(queue_len):
            i, j = queue.pop(0)
            for x, y in [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]:
                if x==row-1 and y==col-1:
                    return res+1
                if 0<=x<row and 0<=y<col and visit[x][y]==0 and matrix[x][y]==0:
                    queue.append([x, y])
                visit[x][y] = 1
        res += 1
    return -1
if n_len:
    print(bfs(matrix))
else:
    print(0)