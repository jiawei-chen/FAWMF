function [ z] = tendot(x,y )
	[n,d]=size(x);
	[n1,k]=size(y);
	if(n~=n1)
		error('n!=n1');
	end

	f=repmat(x,1,k);
	g=reshape(y,n,1,k);
	g=repmat(g,1,d,1);
	g=reshape(g,n,d*k);
	z=f.*g;




end

