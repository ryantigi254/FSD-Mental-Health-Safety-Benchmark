import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import sys
from pathlib import Path

def convert_py_to_ipynb(py_path, ipynb_path):
    print(f"Converting {py_path} -> {ipynb_path}")
    
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    nb = new_notebook()
    cells = []
    
    current_lines = []
    
    # Simple state machine
    # We assume the file is split by "# In[ ]:" markers
    # Blocks before the first marker or between markers need to be analyzed
    
    chunks = []
    current_chunk = []
    
    for line in lines:
        if line.strip().startswith("# In["):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
        else:
            current_chunk.append(line)
            
    if current_chunk:
        chunks.append(current_chunk)
        
    # Process chunks
    for chunk in chunks:
        if not chunk:
            continue
            
        # Analyze chunk to see if it's markdown or code
        # Heuristic: If all non-empty lines start with "#", it's markdown
        # Exception: Shebangs and coding declarations at the start
        
        content = "".join(chunk).strip()
        if not content:
            continue
            
        lines_in_chunk = [l for l in chunk if l.strip()]
        is_markdown = True
        
        # Check against common code indicators
        code_indicators = ["import ", "def ", "class ", "print(", "=", "if ", "for ", "return"]
        
        has_code = False
        for l in lines_in_chunk:
            l_strip = l.strip()
            if l_strip.startswith("#"):
                continue # Comment line
            # Found a line not starting with #, so it's code
            is_markdown = False
            has_code = True
            break
            
        if is_markdown and not has_code:
            # Convert comments to markdown
            md_lines = []
            for l in chunk:
                l_strip = l.strip()
                if l_strip.startswith("#"):
                    # Remove first # and optional space
                     l_content = l_strip[1:]
                     if l_content.startswith(" "):
                         l_content = l_content[1:]
                     md_lines.append(l_content)
                elif not l_strip:
                    md_lines.append("")
            
            cells.append(new_markdown_cell("\n".join(md_lines)))
        else:
            # It's code
            cells.append(new_code_cell("".join(chunk)))
            
    nb.cells = cells
    
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Saved {ipynb_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python py_to_ipynb.py <input.py> <output.ipynb>")
        sys.exit(1)
        
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])

