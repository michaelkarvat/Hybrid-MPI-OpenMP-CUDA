The program receives:
- A matching threshold value
- A list of **Pictures** (large N×N matrices)
- A list of **Objects** (smaller k×k matrices)

Goal:
Check if any object appears inside each picture.

How it works:
- The object is “slid” across the picture like a moving window.
- For each position, a **matching score** is calculated based on the relative difference between matrix values.
- If the score is **below the threshold**, the object is considered **found** in that position.

Output:
For each picture:
- If a matching object is found → print the object ID and the (row, column) position.
- Otherwise → print that no object was found.

Performance:
The search is accelerated using:
- **MPI** to divide pictures across processes
- **OpenMP** to use multiple CPU threads
- **CUDA** to speed up the matching calculation on the GPU
