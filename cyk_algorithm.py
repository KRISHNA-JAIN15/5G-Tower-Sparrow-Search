def cyk_algorithm(grammar, input_string):
    if len(input_string) == 0:
        for var, productions in grammar:
            if var == 'S' and 'ε' in productions:
                return True
        return False
    n = len(input_string)
    table = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for var, productions in grammar:
            for production in productions:
                if len(production) == 1 and production == input_string[i]:
                    table[i][i].add(var)
    for length in range(2, n+1):  
        for i in range(n-length+1):  
            j = i + length - 1  
            for k in range(i, j):  
                for var, productions in grammar:
                    for production in productions:
                        if len(production) == 2:  
                            B, C = production[0], production[1]
                            if B in table[i][k] and C in table[k+1][j]:
                                table[i][j].add(var)
    print("\nParsing Table:")
    for i in range(n):
        for j in range(i, n):
            if table[i][j]:
                print(f"[{i},{j}]: {', '.join(sorted(table[i][j]))}")
    return 'S' in table[0][n-1]
def parse_grammar(grammar_str):
    grammar = []
    lines = grammar_str.strip().split('\n')
    for line in lines:
        if '->' in line:
            var, productions_str = line.split('->')
            var = var.strip()
            productions = [p.strip() for p in productions_str.split('|')]
            grammar.append((var, productions))
    return grammar
def convert_to_cnf(grammar):
    nullable = set()
    for var, productions in grammar:
        if 'ε' in productions:
            nullable.add(var)
    changed = True
    while changed:
        changed = False
        for var, productions in grammar:
            if var not in nullable:
                for production in productions:
                    if all(symbol in nullable for symbol in production) and production != 'ε':
                        nullable.add(var)
                        changed = True
                        break
    cnf_grammar = []
    for var, productions in grammar:
        new_productions = [p for p in productions if p != 'ε']
        for production in productions:
            if production != 'ε':      
                nullable_positions = [i for i, symbol in enumerate(production) if symbol in nullable]    
                for mask in range(1, 2 ** len(nullable_positions)):
                    positions_to_remove = [nullable_positions[i] for i in range(len(nullable_positions)) if (mask & (1 << i))]
                    new_production = ''.join(symbol for i, symbol in enumerate(production) if i not in positions_to_remove)
                    if new_production and new_production not in new_productions:
                        new_productions.append(new_production) 
        if new_productions:
            cnf_grammar.append((var, new_productions))
    return cnf_grammar
def format_grammar(grammar):
    result = []
    for var, productions in grammar:
        result.append(f"{var} -> {' | '.join(productions)}")
    return '\n'.join(result)
grammar_str = """
S -> AB|XB|ε
T -> AB|XB
X -> AT
A -> a
B -> b
"""
grammar = parse_grammar(grammar_str)
cnf_grammar = convert_to_cnf(grammar)
print("Original Grammar:")
print(format_grammar(grammar))
print("\nConverted Grammar:")
print(format_grammar(cnf_grammar))
test_strings = ["aaabb", "aaabbb"]
for test_string in test_strings:
    result = cyk_algorithm(cnf_grammar, test_string)
    print(f"\nIs '{test_string}' in the language? {result}")