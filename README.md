# tetris

# Features

<figure align="center">
<img src="docs/features.png" alt="feature 1" width=""/>
</figure>

### $(f_1)$ : Landing height

The height at which the current piece fell.

### $(f_2)$ : Eroded pieces

The contribution of the last piece to the cleared lines times the number of cleared lines.

### $(f_3)$ : Row transitions

Number of filled cells adjacent to empty cells summed over all rows.

### $(f_4)$ : Column transition

Number of filled cells adjacent to empty cells summed over all columns. Borders count as filled cells.

### $(f_5)$ : Number of holes

The number of empty cells with at least one filled cell above.

### $(f_6)$ : Cumulative wells

The sum of the accumulated depths of the wells.

### $(f_7)$ : Hole depth

The number of filled cells above holes summed over all columns.

### $(f_8)$ : Row hole

The number of rows that contains at least one hole.