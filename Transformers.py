from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('La fracturación hidráulica o fracking es una técnica que permite extraer el llamado gas de esquisto, un tipo de hidrocarburo no convencional que se encuentra literalmente atrapado en capas de roca, a gran profundidad (ver animación arriba y gráfico a la derecha). Luego de perforar hasta alcanzar la roca de esquisto, se inyectan a alta presión grandes cantidades de agua con aditivos químicos y arena para fracturar la roca y liberar el gas, metano. Cuando el gas comienza a fluir de regreso lo hace con parte del fluido inyectado a alta presión. La fracturación hidráulica no es nueva. En el Reino Unido se utiliza para explotar hidrocarburos convencionales desde la década del 50. Pero sólo recientemente el avance de la tecnología y la perforación horizontal permitió la expansión a gran escala del fracking, especialmente en EE.UU., para explotar hidrocarburos no convencionales.')

passage_embedding = model.encode(['La fracturación hidráulica, fractura hidráulica, ​ hidrofracturación​ o simplemente fracturado es una técnica para posibilitar o aumentar la extracción de gas y petroleo del subsuelo, siendo una de las técnicas de estimulación de pozos en yacimientos de hidrocarburos.',
                                  'Es uno de los mejores jugadores de fútbol',
                                  'La inteligencia artificial (IA) hace posible que las máquinas aprendan de la experiencia, se ajusten a nuevas aportaciones y realicen tareas como seres humanos.',
                                  'Es un deportista de alto nivel',
                                  'Barcelona accedió a pagar el tratamiento de la enfermedad hormonal que le habían diagnosticado de niño',])

print("Similarity", util.dot_score(query_embedding, passage_embedding))