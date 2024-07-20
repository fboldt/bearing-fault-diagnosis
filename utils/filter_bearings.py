def filter_bearings(generate_tuples_func, all_strings, any_strings):
    def filtered_bearings():
        # Gerar a lista de tuplas usando a função fornecida
        bearings_list = generate_tuples_func()
        
        # Filtrar a lista de tuplas com base nas strings fornecidas
        filtered_list = [
            bearing for bearing in bearings_list
            if all(string in bearing[0] for string in all_strings) and
                any(string in bearing[0] for string in any_strings)
        ]
        
        return filtered_list
    
    return filtered_bearings