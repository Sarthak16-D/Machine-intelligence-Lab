"""
You can create any other helper funtions.
Do not modify the given functions

"""

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    visited = []
    path.append(start_point)
    f=[[0+heuristic[start_point], path]]
    while (len(f) > 0):
        c, p = f.pop(0)
        n = p[-1]
        c -= heuristic[n]
        if n in goals:
            return p
        visited.append(n)
        t=[]
        for i in range(len(cost[n])):
                if(cost[n][i]>0):
                        t.append(i)
        for i in t:
            P = p + [i]
            C = c + cost[n][i] + heuristic[i]
            if i not in visited and P not in [i[1] for i in f]:
                f.append((C, P))
                f = sorted(f, key=lambda x: (x[0], x[1]))
            elif P in [i[1] for i in f]:
                index = next(c for c in range(len(f)) if f[c][1] == P)
                f[index][0] = min(f[index][0], C)
                f = sorted(f, key=lambda x: (x[0], x[1]))
    return path

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    visited=[0]*len(cost)
    visited[start_point]=1
    s=[]
    s.append(start_point)
    while(len(s)):
        tmp=s.pop()
        path.append(tmp)
        if(tmp in goals):                
                break
        t=[]
        for i in range(len(cost[tmp])):
                if(cost[tmp][i]>0):
                        t.append([i,cost[tmp][i]])
                        if(i in s):
                                s.remove(i)
        t.sort(key = lambda x: x[0])
        t.reverse()
        for j in t:
                if ((j[0] not in path) and (j[0] not in s)):                   
                        s.append(j[0])
    return path

