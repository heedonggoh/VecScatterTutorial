#include <petsc.h>

static char help[] = 
"\n--------------------------------------------------------------------------\n\
A very simple VecScatter Tutorial by Heedong Goh <wellposed@gmail.com> \n\
Performs 'local to global' and 'global to local' scatter operations. \n\
--------------------------------------------------------------------------\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
  PetscErrorCode ierr;
  PetscInt N=5, cpuSize, cpuRank, i;
  PetscScalar sum, ref;
  Vec local,global;
  VecScatter ctx,ctx2;
  IS is;

  ierr = PetscInitialize(&argc,&args,(char*)0,help); CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&cpuSize);   CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&cpuRank);   CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL); CHKERRQ(ierr);

  /* Create and initialize global and local vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&global);    CHKERRQ(ierr);
  ierr = VecSetSizes(global,PETSC_DECIDE,N);     CHKERRQ(ierr);
  ierr = VecSetFromOptions(global);              CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&local); CHKERRQ(ierr);
  ierr = VecSet(local,cpuRank);                  CHKERRQ(ierr);
  for(i=0;i<N;++i) {
    ierr = VecSetValue(local,i,cpuRank*i,INSERT_VALUES); CHKERRQ(ierr);
  }

  /* Add local vectors together into a global vector */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);                     CHKERRQ(ierr);
  ierr = VecScatterCreate(local,is,global,NULL,&ctx2);                  CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx2,local,global,ADD_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx2,local,global,ADD_VALUES,SCATTER_FORWARD);   CHKERRQ(ierr);

  /* Copy the global vector to each local vector */
  ierr = VecScatterCreateToAll(global,&ctx,&local);                       CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,global,local,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,global,local,INSERT_VALUES,SCATTER_FORWARD);   CHKERRQ(ierr);

  /* check */
  ref = 0.0; for(i=0;i<N;++i) ref += i;
  sum = 0.0; for(i=0;i<cpuSize;++i) sum += i;
  ref *= sum;
  ierr = VecNorm(local,NORM_1,&sum); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"cpu %d; ref = %e; L1 norm = %e; error %e\n",cpuRank,ref,sum,fabs(ref-sum)); CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ctx);  CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx2); CHKERRQ(ierr);
  ierr = VecDestroy(&global);      CHKERRQ(ierr);
  ierr = VecDestroy(&local);       CHKERRQ(ierr);
  ierr = ISDestroy(&is);           CHKERRQ(ierr);
  ierr = PetscFinalize();          CHKERRQ(ierr);
  return 0;
}


