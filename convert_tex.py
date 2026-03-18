import os
import re

def convert_tex_to_myst(tex_content):
    content = tex_content
    
    # Remove preamble (keep content after \begin{document})
    if '\\begin{document}' in content:
        content = content.split('\\begin{document}')[1]
    if '\\end{document}' in content:
        content = content.split('\\end{document}')[0]
        
    # Extract Title/Author/Date if present in preamble (usually before document, but sometimes inside)
    # Since we split, title might be lost if it was before. 
    # But usually \title is inside or before.
    # Let's simple-parse the whole file for title to use in metadata, but for content we look at body.
    
    # Basic replacements
    # Sections
    content = re.sub(r'\\section\*?\{([^}]+)\}', r'# \1', content)
    content = re.sub(r'\\subsection\*?\{([^}]+)\}', r'## \1', content)
    content = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'### \1', content)
    
    # Bold / Italic
    content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)
    content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)
    content = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', content)
    
    # Delimiters
    content = re.sub(r'\\bra\{([^}]+)\}', r'\\langle \1 |', content)
    content = re.sub(r'\\ket\{([^}]+)\}', r'| \1 \\rangle', content)
    content = re.sub(r'\\avg\{([^}]+)\}', r'\\langle \1 \\rangle', content)
    content = re.sub(r'\\braket\{([^}]+)\}\{([^}]+)\}', r'\\langle \1 | \2 \\rangle', content)

    # Math
    # Replace \begin{equation} ... \end{equation} with $$ ... $$
    # We need to be careful with multiline. Regex dotall.
    
    def math_replacer(match):
        inner = match.group(1)
        return f'$$\n{inner}\n$$'
    
    content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', math_replacer, content, flags=re.DOTALL)
    
    # Align environments - MyST supports them inside $$
    def align_replacer(match):
        inner = match.group(1)
        return f'$$\n\\begin{{aligned}}\n{inner}\n\\end{{aligned}}\n$$'
        
    # Standard align in latex is often supported by mathjax directly if we wrap in $$
    # But myst/mathjax often needs explicit boundaries.
    content = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', lambda m: f'$$\n\\begin{{align}}{m.group(1)}\\end{{align}}\n$$', content, flags=re.DOTALL)
    
    # Clean up some commands we don't need or can't easily handle
    content = re.sub(r'\\maketitle', '', content)
    content = re.sub(r'\\tableofcontents', '', content)
    content = re.sub(r'\\newpage', '', content)
    content = re.sub(r'\\cancel\{([^}]+)\}', r'\1', content) # Remove cancel strikethrough or support it? Remove for now.
    
    # Citations - leave as is or simple text
    # \cite{...} -> {cite}`...` is myst syntax
    content = re.sub(r'\\cite\{([^}]+)\}', r'{cite}`\1`', content)
    
    return content

def main():
    source_dir = 'theory'
    target_dir = 'theory_web'
    
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.tex')])
    
    for fname in files:
        with open(os.path.join(source_dir, fname), 'r') as f:
            tex_content = f.read()
            
        md_content = convert_tex_to_myst(tex_content)
        
        # Add basic frontmatter
        title = fname.replace('.tex', '')
        # Try to find title in content
        title_match = re.search(r'\\title\{([^}]+)\}', tex_content)
        if title_match:
            title = title_match.group(1)
            
        final_content = f"---\ntitle: {title}\n---\n\n{md_content}"
        
        target_path = os.path.join(target_dir, fname.replace('.tex', '.md'))
        with open(target_path, 'w') as f:
            f.write(final_content)
        print(f"Converted {fname} -> {target_path}")

if __name__ == "__main__":
    main()
