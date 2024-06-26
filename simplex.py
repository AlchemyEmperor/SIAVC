### Author
### Igo Ramalho Brilhante
###
### Simplex : Implementation of Simplex

from tableau import Tableau


# Class to represent Simplex Two Phases method

class Simplex(object):
	"""docstring for Simplex2D"""
	def __init__(self, tableau ):
		super(Simplex, self).__init__()
		self.tableau = tableau

	# First Phase
	def phase1(self):
		# print 'Simplex Phase 1'

		res = True
		if self.checkFeasibility() == False:

			self.addArtificialVariables()
			self.addNewCostFunction()

			# print self.tableau

			simplex = Simplex(self.tableau)
			count = 1
			while simplex.canContinue():
				# print ''
				# print 'Iteration '+str(count)
				# print ''
				simplex.iteration()
				count += 1
				# print simplex.tableau
			# In this case, the Phase 1 has finished with artificial variables
			if len(set(self.tableau.basis) & set(self.tableau.artificial_variable)) > 0:
				# print 'There still exist artifical variables'
				res =  False
			else:
				# The sum of the infeasibilities is greater than 0, what characterizes an unfeasible solution
				if self.tableau[self.tableau.cost_index][self.tableau.b_index] > 0:
					# print 'Sum of artificial variables is greater than 0, then the problem is not feasible'
					self.solution = 'infeasible'
					res = False
				# Everything has happened fine. A feasible tableau has been found to proceed to Phase 2
				else:
					# print 'Phase 1 has found feasible tableau'
					# Remove the artificial cost function
					self.tableau.removeRow(self.tableau.cost_index)
					self.tableau.cost_index = self.tableau.lines -1

					# Remove the artifical variables
					for i in self.tableau.artificial_variable:
						self.tableau.removeColumn(self.tableau.columns-2)

					self.tableau.b_index = self.tableau.columns-1

		# print 'Simplex Phase 1 End'
		return res

	# Second Phase
	# This phase occurs only if the method has found a feasible solution in Phase 1
	def phase2(self):
		# print ''
		# print 'Simplex Phase 2'
		i = 1
		# print '### Initial Tableau Phase 2'
		# print self.tableau
		b = True
		while self.canContinue():
			# print ''
			# print '###### Iteration',i
			# print ''
			b = self.iteration()
			# print self.tableau
			if b == False:
				break
			i += 1
		if b == True:
			self.solution = self.tableau[self.tableau.cost_index][self.tableau.b_index]
		# print ''
		# print 'Simplex Phase 2 End'

	# Exceute the method
	def execute(self):
		self.solution = None
		
		r = self.phase1()
		
		if r == True:
			self.phase2()

			

	# Add artificial variables into Tableau
	def requeredArtificalVariables(self):
		n = self.tableau.constraints_count
		n_of_variables = 0
		c = list()
		for i in range(0,n):
			for j in range(self.tableau.var_count,self.tableau.columns):
				if j - self.tableau.var_count == i:
					if self.tableau[i][j] != 1:
						n_of_variables += 1
						c.append(i)

		return n_of_variables,c

	# Add artificial variable during the Phase 1	
	def addArtificialVariables(self):
		n_of_variables,c = self.requeredArtificalVariables()

		if n_of_variables  > 0:

			self.tableau.artificial_variable_count = n_of_variables
			for i in range(0,n_of_variables):
				idx = self.tableau.columns - 1
				self.tableau.addColumn(idx,0.0)
			
			idx = self.tableau.var_count+self.tableau.constraints_count
			for r in c:
				self.tableau.artificial_variable.append(idx)
				self.tableau[r][idx] = 1.0

				self.tableau.basis.remove(r+self.tableau.var_count)
				self.tableau.basis.append(idx)

				idx += 1

		self.tableau.b_index = self.tableau.columns - 1

	# Add an artificial cost function W during the Phase 1
	def addNewCostFunction(self):
		

		n_of_variables,c = self.requeredArtificalVariables()

		if n_of_variables > 0:
			self.tableau.addRow(self.tableau.lines,0.0)
			idx = self.tableau.lines - 1
			for a in self.tableau.artificial_variable:
				self.tableau[idx][a] = 1.0

			for r in c:
				for i in range(0,self.tableau.columns):
					self.tableau[idx][i] = self.tableau[idx][i] -  self.tableau[r][i]

			self.tableau.cost_index = self.tableau.lines - 1

	# Check if there still exist a direction of decreasing
	def canContinue(self):
		cost_index = self.tableau.cost_index
		for i in range (0,self.tableau.columns):
			if self.tableau[cost_index][i] < 0:
				return True

		return False

	# def phase1(self):

	def checkFeasibility(self):
		n,c = self.requeredArtificalVariables()
		# print n
		if n == 0:
			return True
		return False




	# Get the pivot, which is the column with the lowest value in cost line
	def getPivot(self):
		cost_index = self.tableau.cost_index
		pivot = 0
		for i in range (0,self.tableau.columns-1):
			if self.tableau[cost_index][i] < pivot:
				pivot = self.tableau[cost_index][i]

		return self.tableau[cost_index].index(pivot)

	# Check if the solution is unbounded
	def isBoundedSolution(self,pivot):
		
		limit = float("inf")
		line_index = -1
		count = 0
		# Given a pivot, check if there is at least one element in the pivot column that is positive
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0 :
				return True
		return False

	def isDegenerative(self,pivot):
		b_index = self.tableau.b_index

		limit_set  = set()

		limit = float("inf")
		line_index = -1
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0 :
				limit = self.tableau[i][b_index]/self.tableau[i][pivot]

				if limit in limit_set:
					return True

		return False

	# Look for the minimum limit for the variable
	# pivot: pivot index
	def getConstraintLimit(self,pivot):
		b_index = self.tableau.b_index
		# print 'b_index: '+str(b_index)
		limit = float("inf")
		line_index = -1
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0:
				# print 'l : '+str(self.tableau[i][b_index]/self.tableau[i][pivot])
				if self.tableau[i][b_index]/self.tableau[i][pivot] < limit:
					limit = self.tableau[i][b_index]/self.tableau[i][pivot]
					line_index = i
		# print 'Limit %f in %d' %(limit,line_index)
		return line_index

	# Scaling matrix in order to obtain 0 in position 'pivot'
	# i: constraint index
	# pivot: pivot index
	def scalingMatrix(self,i,pivot):
		for j in range(0,self.tableau.lines):
			if i != j:
				# print self.tableau[j]
				pivot_value = self.tableau[j][pivot]
				for k in range(0,self.tableau.columns):
					# print '%f - (%f * %f)' % (self.tableau[j][k],self.tableau[i][k],pivot_value)
					self.tableau[j][k] = self.tableau[j][k] - (self.tableau[i][k]*pivot_value)

	def gaussianOperation(self,constraint_index,pivot_index):

		pivot_value = self.tableau[constraint_index][pivot_index]

		# print 'Pivot Index',pivot_index
		# print 'Pivot Value',pivot_value
		# print 'Constraint Index',constraint_index

		for i in range(0,self.tableau.columns):
			self.tableau[constraint_index][i] = self.tableau[constraint_index][i] / pivot_value

		# print 'Constraint',self.tableau[constaint_index]
		self.scalingMatrix(constraint_index,pivot_index)

	# Perform iteration for Simplex Method
	def iteration(self):
		pivot_index  = self.getPivot()

		if(self.isDegenerative(pivot_index)):
			# print 'Problem is degenerative and it needs to use another method'
			self.solution = 'degenerative'
			return False

		# Check if the solution if bounded
		if(self.isBoundedSolution(pivot_index)):

			constraint_index = self.getConstraintLimit(pivot_index)
			if constraint_index == -1:
				return
			# change basis
			self.tableau.changeBasis(pivot_index,constraint_index)
			# print 'Basis: '+str(self.tableau.basis)

			self.gaussianOperation(constraint_index,pivot_index)
			
			return True
		else:
			# print 'Solution is unbounded'
			self.solution = 'unbounded'
			return False




		