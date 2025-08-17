classdef simul3DcoCyl_cuda_pgse_bvec < handle
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School
    
    properties (GetAccess = public, SetAccess = public)
        TD;
        dtime; Tstep; NPar; Din; pinit;
        rCir; lm; Nm; res;
        lz; Nz; xCir; yCir;
        kappa;
        dx2t; dx4t; sig;
        Del; del; bval; bvec; NDel; Nbtab;
        NParBin; NParICS;
    end
    
    properties (GetAccess = private, SetAccess = private)
        D_cnt = [1 2 2 1 2 1];
        W_cnt = [1 4 4 6 12 6 4 12 12 4 1 4 6 4 1];
        D_ind = [1 1; 1 2; 1 3; 2 2; 2 3; 3 3];
        W_ind = [1 1 1 1; 1 1 1 2; 1 1 1 3; 1 1 2 2; 1 1 2 3;
                1 1 3 3; 1 2 2 2; 1 2 2 3; 1 2 3 3; 1 3 3 3;
                2 2 2 2; 2 2 2 3; 2 2 3 3; 2 3 3 3; 3 3 3 3];
    end
    
    methods (Access = public)
        function this = simul3DcoCyl_cuda_pgse_bvec(root,varargin)
            if nargin < 2
                this.readJob(root);
            else
                this.readJob(root, varargin{1});
            end
        end
        
        function this = readJob(this,root,varargin)
            % Load data
            param = load(fullfile(root,'sim_para.txt'));
            this.dtime = param(1);
            this.Tstep = param(2);
            this.NPar = param(3);
            this.Din = param(4);
            if nargin>2 & strcmpi(varargin{1},'bead')
                this.lm = param(5)*param(9);
                this.Nm = param(6);
                this.lz = param(7)*param(9);
                this.Nz = param(8);
                this.res = param(9);
                this.rCir = load(fullfile(root,'phantom_rCir.txt'))*param(9);
                this.xCir = load(fullfile(root,'phantom_xCir.txt'))*param(9);
                this.yCir = load(fullfile(root,'phantom_yCir.txt'))*param(9);
            else
                this.rCir = param(5)*param(8);
                this.lm   = param(6)*param(8);
                this.Nm   = param(7);
                this.res  = param(8);
                try
                    this.kappa = param(9);
                end
            end

            this.TD = load(fullfile(root,'diff_time.txt'));
            
            this.dx2t = load(fullfile(root,'dx2_diffusion.txt')).*this.D_cnt/this.NPar;
            this.dx4t = load(fullfile(root,'dx4_diffusion.txt')).*this.W_cnt/this.NPar;
            
            DELdel = load(fullfile(root,'gradient_DELdel.txt'));
            DELdel = reshape(DELdel,2,[]).';
            this.Del = DELdel(:,1);
            this.del = DELdel(:,2);
            this.NDel = numel(this.Del);

            btab = load(fullfile(root,'gradient_btab.txt'));
            btab = reshape(btab,4,[]).';
            this.bval = btab(:,1);
            this.bvec = btab(:,2:4);
            this.Nbtab = numel(this.bval);

            this.sig = load(fullfile(root,'sig_diffusion.txt'))/this.NPar;
            this.sig = reshape(this.sig,this.Nbtab,this.NDel);

            % try
                this.NParBin = load(fullfile(root,'NParBin.txt'));
                this.NParICS = load(fullfile(root,'NParICS.txt'));
            % end
            
        end
        
        function dt = dki_shell(this)
            Nt = size(this.sig,1);
            bvecu = unique(this.bvec,'row');
            n2 = this.ndir2(bvecu);
            n4 = this.ndir4(bvecu);
            
            nbvec = size(bvecu,1);
            nbval = numel(unique(this.bval));
            dt = zeros(Nt,21);
            for i = 1:Nt
                sigi = abs(this.sig(i,:));
                Di = zeros(nbvec,1);
                Ki = zeros(nbvec,1);
                for j = 1:nbvec
                    Ij = ismember(this.bvec,bvecu(j,:),'rows');
                    sigj = sigi(Ij); sigj = sigj(:);
                    bvalj = this.bval(Ij); bvalj = bvalj(:);
                    A = [-bvalj 1/6*bvalj.^(2:nbval)];
                    X = A\log(sigj + eps);
                    Di(j) = X(1);
                    Ki(j) = X(2)/X(1)^2;
                end
                dx2g = Di*2.*this.TD(i);
                dx4g = (Ki+3).*dx2g.^2;
                dt(i,1:6) = ( n2\dx2g ).';
                dt(i,7:21) = ( n4\dx4g ).';
            end
        end
                
        function [K,D] = akc_mom(this,n)
            n2 = this.ndir2(n);
            n4 = this.ndir4(n);
            x2 = this.dx2t*n2.';
            x4 = this.dx4t*n4.';
            D = x2/2./this.TD;
            K = x4./x2.^2-3;
        end
        
        function [K,D] = akc_shell(this,dt,n)
            n2 = this.ndir2(n);
            n4 = this.ndir4(n);
            x2 = dt(:,1:6)*n2.';
            x4 = dt(:,7:21)*n4.';
            D = x2/2./this.TD;
            K = x4./x2.^2-3;
        end
        
        function n4i = ndir4(this,ni)
            n4i = prod(reshape(ni(:,this.W_ind),[],15,4),3);
        end
        
        function n2i = ndir2(this,ni)
            n2i = prod(reshape(ni(:,this.D_ind),[],6,2),3);
        end
    end
    
        
end