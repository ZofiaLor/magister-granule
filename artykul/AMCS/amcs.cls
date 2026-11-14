%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------LaTeX-DOCUMENT-CLASS-----------%
%------------------------------------------%
%--------for-preparing-papers-to-the-------%
%------------------------------------------%
%---------INTERNATIONAL-JOURNAL-OF---------%
%-APPLIED-MATHEMATICS-AND-COMPUTER-SCIENCE-%
%------------------------------------------%
%----------A------MM---MM----CCCC----SSSS--%
%--------A-A-----M-M-M-M---C----C--S-------%
%------AAAAA----M--M--M---C--- ----SSSS----%
%----A-----A---M-----M---C----C-------S----%
%--A-------A--M-----M----CCCC----SSSS------%
%------------------------------------------%
%------------------------------------------%
%------Zielona Gora,-10-December-2016------%
%------------------------------------------%
%-----------(c)-Krzysztof-Patan------------%
%---------e-mail:-amcs@uz.zgora.pl---------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% e-mail address change (10.12.2016) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Document Style `amcs' <2016>.}

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{amcs}[2016/12/10]
\LoadClass[twoside,twocolumn,a4paper]{article}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New option editorial added 14.12.2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption{editorial}{\@twocolumnfalse}
\ProcessOptions

\RequirePackage{times}
\RequirePackage{amsmath,amssymb}
\RequirePackage[hang]{caption2}
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{ifthen}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This definition is needed due to problems with the
%% hyperref package and its compatibility with harvard
%% in the newest versions of MiKTeX (4.11.2010)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\IfFileExists{ltxcmds.sty}{\RequirePackage{ltxcmds}}{}
\long\def\ltx@LocalAppendToMacro#1#2{%
\global\let\LTXcmds@gtemp#1%
\ifx\ltx@undefined\LTXcmds@gtemp
\global\let\LTXcmds@gtemp\ltx@empty
\else
\ifx\relax\LTXcmds@gtemp
\global\letLTXcmds@gtemp\ltx@empty
\fi
\fi
\begingroup
\toks0\expandafter{\LTXcmds@gtemp#2}%
\xdef\LTXcmds@gtemp{\the\toks0}%
\endgroup
\let#1\LTXcmds@gtemp
}

\RequirePackage[dcucite,abbr]{harvard}
\RequirePackage{url}
%Added 15.05.2017
\RequirePackage{hyperref}

%% Added 24.01.2012
\RequirePackage{algorithm}
\renewcommand\floatc@ruled[2]{{\@fs@cfont #1.} #2\par}
\RequirePackage{algorithmic}

%%%%%%%%%%%%%%%%%%%%%%%%% Page layout %%%%%%%%%%%%%%%%%%%%%%%%
%% Last change 26.03.2010
\marginparsep=.1cm
\setlength{\textheight}{23.2cm}
\setlength{\textwidth}{17cm}
\addtolength{\oddsidemargin}{-0.cm}
\addtolength{\evensidemargin}{-1cm}
\setlength{\headsep}{4mm}
\addtolength{\topmargin}{-.2cm}
\columnsep 8mm
\linespread{1}
\setlength{\parindent}{7mm}

%%%%%%%%%%%%%%%%%%%%%%% Editorial variables %%%%%%%%%%%%%%%%%%%%
\newcommand{\Year}[1]{\gdef\@Year{#1}}
\newcommand{\Vol}[1]{\gdef\@Vol{#1}}
\newcommand{\No}[1]{\gdef\@No{#1}}
\newcommand{\Startpage}[1]{\gdef\@Startpage{#1}}
\newcommand{\Endpage}[1]{\gdef\@Endpage{#1}}
\newcommand{\Runauthors}[1]{\gdef\@Runauthors{#1}}
\newcommand{\Received}[1]{\gdef\@Received{#1}}
\newcommand{\Revised}[1]{\gdef\@Revised{#1}}
\newcommand{\Rerevised}[1]{\gdef\@Rerevised{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 9.05.2016: Added commands \Accepted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\Accepted}[1]{\gdef\@Accepted{#1}} 
\newcommand{\DOI}[1]{\gdef\@DOI{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 22.10.2014: Added commands \authors, \addresses 
%% and \correspondingauthor 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\authors}[1]{\gdef\@authors{#1}}
\newcommand{\addresses}[1]{\gdef\@addresses{#1}}

\newcommand{\correspondingauthor}[1]{\edef\@corresponding{#1}}

\let\@Year\@empty
\let\@Vol\@empty
\let\@No\@empty
\let\@Startpage\@empty
\let\@Endpage\@empty
\let\@Runauthors\@empty
\let\@Received\@empty
\let\@Revised\@empty
\let\@Rerevised\@empty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6.05.2016: Added variable setting for \@accepted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\@Accepted\@empty
\let\@DOI\@empty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 22.10.2014: Added variables setting for \@authors, 
%%\@addresses and \@corresponding 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\@authors\@empty
\let\@addresses\@empty
\let\@corresponding\@empty

%%%%%%%%%%%%%%%%%%%%%%%% Title page %%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\title}[2][]{
 \def\@temp{#1}
 \protected@edef\@title{#2} %Redefined 29.06.016
 \ifx\@temp\@empty
   \gdef\head@title{#2}
 \else
   \gdef\head@title{#1}
 \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% 22.10.2014: Changed list of markers 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\gdef\@znak#1{\ifcase#1 \or {a} \or {b} \or {c} \or {d} \or {e} \or {f}\fi}

\let\@authora\@empty\let\@authorb\@empty
\let\@authorc\@empty\let\@authord\@empty
\let\@authore\@empty\let\@authorf\@empty

\let\@markaa\@empty\let\@markba\@empty\let\@markca\@empty
\let\@markda\@empty\let\@markea\@empty\let\@markfa\@empty

\let\@markab\@empty\let\@markbb\@empty\let\@markcb\@empty
\let\@markdb\@empty\let\@markeb\@empty\let\@markfb\@empty

\gdef\@autor#1#2{\ifcase#1\or\gdef\@authora{#2}\or\gdef\@authorb{#2}\or\gdef\@authorc{#2}
   \or\gdef\@authord{#2}\or\gdef\@authore{#2}\or\gdef\@authorf{#2}\fi}

\gdef\@autoradra#1#2{\ifcase#1\or\gdef\@markaa{#2}\or\gdef\@markba{#2}\or\gdef\@markca{#2}
   \or\gdef\@markda{#2}\or\gdef\@markea{#2}\or\gdef\@markfa{#2}\fi}

\gdef\@autoradrb#1#2{\ifcase#1\or\gdef\@markab{#2}\or\gdef\@markbb{#2}\or\gdef\@markcb{#2}
   \or\gdef\@markdb{#2}\or\gdef\@markeb{#2}\or\gdef\@markfb{#2}\fi}

\newcounter{author}
\def\author[#1][#2]#3{
 \stepcounter{author}
 \@autor{\theauthor}{#3}
 \def\@temp{#1}
 \ifx\@temp\@empty
   \errmessage{\@temp: Institution not defined}
 \else
   \@autoradra{\theauthor}{#1}
   \@autoradrb{\theauthor}{#2}	
 \fi
 }

\let\@adr\@empty
\let\@adra\@empty\let\@adrb\@empty\let\@adrc\@empty
\let\@adrd\@empty\let\@adre\@empty\let\@adrf\@empty

\let\@markad\@empty\let\@markbd\@empty\let\@markcd\@empty
\let\@markdd\@empty\let\@marked\@empty\let\@markfd\@empty

\gdef\@adres#1#2#3{\ifcase#1\or\gdef\@adra{#2}\gdef\@markad{#3}\or\gdef\@adrb{#2}\gdef\@markbd{#3}\or\gdef\@adrc{#2}\gdef\@markcd{#3}\or\gdef\@adrd{#2}\gdef\@markdd{#3}\or\gdef\@adre{#2}\gdef\@marked{#3}\or\gdef\@adrf{#2}\gdef\@markfd{#3}\fi}

\gdef\@sprawdza#1{\ifcase#1\or\@markaa\or\@markba\or\@markca\or\@markda\or\@markea\or\@markfa\fi}
\gdef\@sprawdzb#1{\ifcase#1\or\@markab\or\@markbb\or\@markcb\or\@markdb\or\@markeb\or\@markfb\fi}
\gdef\@sprawdzautor#1{\ifcase#1\or\@authora\or\@authorb\or\@authorc\or\@authord\or\@authore\or\@authorf\fi}
\gdef\@sprawdzadres#1{\ifcase#1\or\@markad\or\@markbd\or\@markcd\or\@markdd\or\@marked\or\@markfd\fi}
\gdef\@sprawdzc#1{\ifcase#1\or\@adra\or\@adrb\or\@adrc\or\@adrd\or\@adre\or\@adrf\fi}

\newcounter{address}
\newcommand{\address}[2][]{
 \stepcounter{address}
 \edef\@temp{#1}
 \ifx \@temp\@empty
 \else
  \@adres{\theaddress}{#2}{#1}
  \count0=1\relax
  \loop
    \ifnum \theaddress>\count0
      \edef\@tempa{\@sprawdzadres{\the\count0}}
      \ifx \@temp\@tempa \errmessage{\@temp: Multiple definition} \fi
    \fi
  \advance\count0by1
  \ifnum \count0<\theaddress\repeat
 \fi
}

\def\test@autor{\count0=6\relax
  \loop
   \edef\@temp{\@sprawdza{\the\count0}}
   \ifx\@temp\@markad \else
    \ifx\@temp\@markbd \else
     \ifx\@temp\@markcd \else
      \ifx\@temp\@markdd \else
       \ifx\@temp\@marked \else
        \ifx\@temp\@markfd \else
	 \@autoradra{\the\count0}{\@empty}
	 \errmessage{\@temp: Institution not defined}
   \fi\fi\fi\fi\fi\fi
   \advance\count0by-1
  \ifnum \count0>0  \repeat
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 26.03.2010: New definitions of formatting 
%% authors and addresses in the frontmatter
%% 22.10.2014: Modified markers arragement 
%% for addresses of authors
%% 22.10.2014: Added marker for corresponding 
%% author   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@format@autor{
 \count1=1\relax
 \count2=1\relax
 \edef\@tempa{\@sprawdza{1}}
 \edef\@tempb{\@sprawdzb{1}}
 \count0=2\relax\loop
    \edef\@tempc{\@sprawdza{\the\count0}}
    \edef\@tempd{\@sprawdzb{\the\count0}} 
    \ifx\@tempa\@tempc 
        \advance\count1by1\relax
    \fi
    \ifx\@tempb\@tempd 
        \advance\count2by1\relax
    \fi
 \advance\count0by1\ifnum\count0<\theauthor\repeat
 %\ifnum\the\count1=\the\count2
 %   \count3=1\relax\loop
 %   \ifnum\the\count3=1
 %     \@sprawdzautor{\the\count3}
 %   \else  
 %    , \mbox{\@sprawdzautor{\the\count3}}
 %   \fi
 %   \advance\count3by1\ifnum\count3<\theauthor\repeat
 %\else
   \count2=\theauthor \advance\count2by1
   \count0=1\relax
   \loop
    \edef\@tempa{\@sprawdzautor{\the\count0}}\relax
    \edef\@tempb{\@sprawdza{\the\count0}}\relax    
    \edef\@tempbdwa{\@sprawdzb{\the\count0}}\relax
    \ifx \@tempb\@empty
      \mbox{\@tempa}
    \else
      {\count1=1\relax\loop
         \edef\@tempc{\@sprawdzadres{\the\count1}}\relax
         \ifx\@tempc\@tempb
	  %\ifnum\theauthor=1 \mbox{\@tempa}
	  %\else
            %\ifnum\theaddress=1
	    %  \ifnum\theauthor=\the\count0
	    %    \mbox{\@tempa}
	    %  \else \mbox{\@tempa,}
            %  \fi
	    %\else
              \ifx \@tempbdwa\@empty           
	      \ifnum\theauthor=\the\count0
               \ifx\@corresponding\@tempa
                  \mbox{\@tempa\,$^{\@znak{\the\count1},}\footnote{Corresponding author}$}
                \else
 	          \mbox{\@tempa\,$^\@znak{\the\count1}$}
                 \fi 
	      \else 
                \ifx\@corresponding\@tempa
                  \mbox{\@tempa\,$^{\@znak{\the\count1},}\footnote{Corresponding author}$\, ,}
                \else
                  \mbox{\@tempa\,$^\@znak{\the\count1}$,}
                \fi
              \fi
	      \else
	        {\count2=1\relax\loop
	          \edef\@tempcdwa{\@sprawdzadres{\the\count2}}\relax
		  \ifx\@tempcdwa\@tempbdwa
	            \ifnum\theauthor=\the\count0
                      \ifx\@corresponding\@tempa
                         \mbox{\@tempa\,$^\@znak{\the\count1}$$^{\!,\@znak{\the\count2},}\footnote{Corresponding author}$}
                      \else
                         \mbox{\@tempa\,$^\@znak{\the\count1}$$^{\!,\@znak{\the\count2}}$} 
                      \fi
	            \else 
                      \ifx\@corresponding\@tempa
                         \mbox{\@tempa\,$^\@znak{\the\count1}$$^{\!,\@znak{\the\count2},}\footnote{Corresponding author}$\, ,}
                      \else
                         \mbox{\@tempa\,$^\@znak{\the\count1}$$^{\!,\@znak{\the\count2}}$,}
                      \fi
                    \fi
		  \fi
	        \advance\count2by1\ifnum\count2<7\repeat}
	      \fi
	    %\fi
	  %\fi
         \fi
       \advance\count1by1\ifnum\count1<7\repeat}
    \fi
     \advance\count0by1
  \ifnum\count0<\count2\repeat
 %\fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Modified markers arrangement for 
%% addresses of authors 22.10.2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@format@adres{
 \count1=1\relax
 \count2=1\relax
 \edef\@tempa{\@sprawdza{1}}
 \edef\@tempb{\@sprawdzb{1}}
 \count0=2\relax\loop
    \edef\@tempc{\@sprawdza{\the\count0}}
    \edef\@tempd{\@sprawdzb{\the\count0}} 
    \ifx\@tempa\@tempc 
        \advance\count1by1\relax
    \fi
    \ifx\@tempb\@tempd 
        \advance\count2by1\relax
    \fi
 \advance\count0by1\ifnum\count0<\theauthor\repeat
 %\ifnum\the\count1=\the\count2
 %   \@adra
 %   \vskip10pt
 %   \@adrb
 %\else
   %\ifx\@adra\@empty\relax
   %\else
     %\ifnum \theauthor=1 \@adra
     %\else
     %  \ifnum\theaddress=1 \@adra
     %  \else
     %   $^{a}$\@adra 
     %  \fi
     %\fi
   %\fi
 \count1=1\relax\loop
%  \count1=1\relax\loop
   \edef\@tempa{\@sprawdzadres{\the\count1}}\relax
    \def\@tempb{\@sprawdzc{\the\count1}}\relax
    \ifx\@tempa\@empty\relax
    \else
       {\count2=1\relax\loop
        \edef\@tempc{\@sprawdza{\the\count2}}\relax
        \edef\@tempd{\@sprawdzb{\the\count2}}\relax
        \ifx\@tempa\@tempc
           \vskip10pt
           %\ifnum \theauthor=1 \@tempb
           %\else 
            $^\@znak{\the\count1}$\@tempb
           %\fi
           \count2=6
        \else
           \ifx\@tempa\@tempd
             \vskip10pt
             %\ifnum \theauthor=1 \@tempb
             %\else 
             $^\@znak{\the\count1}$\@tempb
             %\fi
             \count2=6
           \else
              \relax
           \fi
         \fi
       \advance\count2by1\ifnum\count2<7\repeat}
    \fi
  \advance\count1by1\ifnum\count1<7\repeat
%\fi
}

\newbox\abstractbox
\def\abstract{\global\setbox\abstractbox=\hbox\bgroup
  \small \begin{minipage}{155mm}}
\def\endabstract{\end{minipage}\egroup}

\newbox\keywordbox
\def\keywords{\global\setbox\keywordbox=\hbox\bgroup
   \small \begin{minipage}[t]{155mm}{\bf Keywords:}}
\def\endkeywords{\end{minipage}\egroup}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% 29.07.2016: redefinition of the harvardurl 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\harvardurl}{URL:\url}

%%%%%%%%%%%%%%%%%%%%%%%% Headers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@titlepage

\def\ps@plain{
 \let\@mkboth\markboth
 \def\@oddhead{}
 \def\@evenhead{}
 \def\@oddfoot{}
 \def\@evenfoot{}
}

\def\ps@amcs{
 \let\@mkboth\markboth
 \let\@oddfoot\@empty
 \let\@evenfoot\@empty
 \def\@oddhead{\vtop{\vskip-5.8em
   \includegraphics[width=18.4cm]{logop}\vskip-2.1em
   \hb@xt@\textwidth{
   \hspace{-0.7mm}\mbox{\raisebox{.5mm}{\sl \head@title}}\hfil
   \parbox{9mm}{\vskip5pt \centering \textcolor{white}{\bf \thepage}}\parbox{2mm}{~}}}}
 \def\@evenhead{\vtop{\vskip-5.8em
   \hspace{-1.4cm}\includegraphics[width=18.4cm]{logol}\vskip-2.1em
   \hb@xt@\textwidth{
    \hspace{0.5mm}\parbox{9mm}{\vskip5pt \centering \textcolor{white}{\bf \thepage}}
   \hfil{\sl \@Runauthors}\hspace{0.5mm}}}}
}
\if@twocolumn
\pagestyle{amcs}
\else
\pagestyle{empty}
\fi
%%%%%%%%%%%%%%%%%%%%%%%% Maketitle %%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% 22.10.2014: \authors and \addresses added to frontmatter as option 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@maketitle{\newpage
      \if@twocolumn
      \vskip4.5mm\vskip-8.3em
      \includegraphics[width=18.4cm]{logop}
      \vskip-3.7em
      {\sl Int. J. Appl. Math. Comput. Sci., \@Year, Vol. \@Vol, No. \@No, \@Startpage--\@Endpage}
      \ifx\@Startpage\@empty\setcounter{page}{1}\else\setcounter{page}{\@Startpage}\fi\newline
      {DOI:~\@DOI}
      \vskip2.2em
      \begin{center}
      \begin{minipage}{155mm}
        \centering
        \vskip42pt
        \ifx\@authors\@empty
            \let\footnote\thanks
	    \let\author\relax
	    \test@autor
	    {\large \bfseries \MakeUppercase{\@title} \par}
	    \vskip20pt
	    {\sc\@format@autor}
	    \vskip20pt
	    {\small \@format@adres}
        \else
            \let\footnote\thanks
	    {\large \bfseries \MakeUppercase{\@title} \par}
	    \vskip20pt
	    {\sc\@authors}
	    \vskip20pt
	    {\small \@addresses}
        \fi	
        \vskip30pt
	\usebox{\abstractbox}
	\vskip20pt
	\usebox{\keywordbox}
        %\footnote{}{Corresponding author}
      \end{minipage}	
      \end{center}\par\vskip35pt
      \else\relax
      \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sections %%%%%%%%%%%%%%%%%%%%%%%%%%
%% Changed 24.01.2012 -- no hyphenation in section names
\if@twocolumn
 \def\section{\@startsection{section}{1}{\z@}{-3.1ex plus -1ex minus
 -.2ex}{1.3ex plus .2ex}{\hyphenpenalty=10000\hyphenchar\font=-1\sloppy\flushleft\large\bf}}
 \renewcommand\@seccntformat[1]{\csname the#1\endcsname. \hspace{0.5em}}
\else
  \newcommand\additem[2]{%
    \def\@temp{#2}
    \addpenalty\@secpenalty
    \ifx\@temp\@empty
      \addvspace{2.25em \@plus\p@}%
      \setlength\@tempdima{2em}%
    \else
      \addvspace{1.0em \@plus\p@}%
      \setlength\@tempdima{3.6em}%
    \fi
    \setlength\@tempdimb{1.8em}%
    \begingroup
      \parindent \z@
      \rightskip 2.5em
      \parfillskip -2.5em
      \leavevmode
      \advance\leftskip\@tempdima
      \hskip -\@tempdimb
      \ifx\@temp\@empty
        #1\nobreak\hfill\nobreak\par
      \else
        #1\nobreak\dotfill~~~\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \fi
    \endgroup
   }
\fi
 
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {-1em}%
                                     {\normalfont\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {-1em}%
                                     {\normalfont\bfseries}}

\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M #8 \@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8.}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}
  
%%%%%%%%%%%%%%%%%%%%% Figures and tables %%%%%%%%%%%%%%%%%%%%%%%%
\def\fnum@figure{Fig.~\thefigure}
\def\fnum@table{Table~\thetable}

\renewcommand{\captionsize}{\small}
\renewcommand{\captionlabeldelim}{.\hspace*{1mm}}



%%%%%%%%%%%%%%%%%%%%%%%%%% Bibliography %%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\refname}{References}
\renewenvironment{thebibliography}[1]
     {\small\section*{\centering \refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
      
%%%%%%%%%%%%%%%%%%%%%%%%% Final information %%%%%%%%%%%%%%%%%%%
\newlength{\dlugosc}
\def\makeinfo{\small
      \vskip10pt
      \flushright
      \ifx\@Revised\@empty
        \setbox\@tempboxa\hbox{Received: \@Received}
	\setlength\dlugosc{\wd\@tempboxa}
        \setbox\@tempboxa\hbox{Accepted: \@Accepted}  % Added 6.05.2016
	\ifdim \dlugosc < \wd\@tempboxa \setlength\dlugosc{\wd\@tempboxa}\fi % Added 6.05.2016	
        \parbox{\dlugosc}{
	Received: \@Received
        \vskip1pt             % Added 6.05.2016
        Accepted: \@Accepted} % Added 6.05.2016
      \else
       \ifx\@Rerevised\@empty
         \setbox\@tempboxa\hbox{Received: \@Received}
	 \setlength\dlugosc{\wd\@tempboxa}
	 \setbox\@tempboxa\hbox{Revised: \@Revised}
	 \ifdim \dlugosc < \wd\@tempboxa \setlength\dlugosc{\wd\@tempboxa}\fi
         \setbox\@tempboxa\hbox{Accepted: \@Accepted}  % Added 6.05.2016
	 \ifdim \dlugosc < \wd\@tempboxa \setlength\dlugosc{\wd\@tempboxa}\fi % Added 6.05.2016
	 \parbox{\dlugosc}{
	 Received: \@Received
         \vskip1pt
         Revised: \@Revised
         \vskip1pt             % Added 6.05.2016
         Accepted: \@Accepted} % Added 6.05.2016
	\else
         \setbox\@tempboxa\hbox{Received: \@Received}
	 \setlength\dlugosc{\wd\@tempboxa}
	 \setbox\@tempboxa\hbox{Revised: \@Revised}
	 \ifdim \dlugosc < \wd\@tempboxa  \setlength\dlugosc{\wd\@tempboxa}\fi
	 \setbox\@tempboxa\hbox{Re-revised: \@Rerevised}
	 \ifdim \dlugosc < \wd\@tempboxa \setlength\dlugosc{\wd\@tempboxa}\fi
	 \setbox\@tempboxa\hbox{Accepted: \@Accepted}  % Added 6.05.2016
	 \ifdim \dlugosc < \wd\@tempboxa \setlength\dlugosc{\wd\@tempboxa}\fi % Added 6.05.2016
	 \parbox{\dlugosc}{Received: \@Received
         \vskip1pt
         Revised: \@Revised
      	 \vskip1pt
         Re-revised: \@Rerevised
         \vskip1pt             % Added 6.05.2016
         Accepted: \@Accepted} % Added 6.05.2016
        \fi
      \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%% Theorem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\partopsep\z@
\parsep\z@
\newcounter{theorem}
\def\theorem#1{
 \refstepcounter{theorem}
 \trivlist\item
 \noindent\textbf{Theorem \thetheorem .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1)\fi
 \it
}
\def\endtheorem{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Proposition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{proposition}
\def\proposition#1{
 \refstepcounter{proposition}
 \trivlist\item
 \noindent\textbf{Proposition \theproposition .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1) \fi
 \it
}
\def\endproposition{\endtrivlist\unskip}
 
%%%%%%%%%%%%%%%%%%%%%%%%%% Lemma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{lemma}
\def\lemma#1{
 \refstepcounter{lemma}
 \trivlist\item
 \noindent\textbf{Lemma~\thelemma .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1) \fi
 \it
}
\def\endlemma{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Corollary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{corollary}
\def\corollary#1{
 \refstepcounter{corollary}
 \trivlist\item
 \noindent\textbf{Corollary \thecorollary .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1) \fi
 \it
}
\def\endcorollary{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Proof %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modified 12.11.2012
%% proof name has been added
\def\proof#1{
 \trivlist\item
 \noindent
 \textit{Proof.}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1) \fi
}
\def\endproof{\hspace*{\fill}\quad{\footnotesize $\blacksquare$}\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modified 12.11.2012
%% example name has been added as the second parameter
\newcounter{example}
\def\example[#1]#2{
 \refstepcounter{example}
 \trivlist\item
 \noindent
 \def\@tempa{#1}\relax
 \def\@tempb{#2}\relax
 \textbf{Example \theexample .}
 \ifx \@tempb\@empty \else (\textit{#2}) \fi
}
\def\endexample{
 \ifx\@tempa\@empty \hspace*{\fill}\quad $\blacklozenge$ \fi
 \endtrivlist\unskip
}

%%%%%%%%%%%%%%%%%%%%%%%%%% Remark %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modified 12.11.2012
%% remark name has been added
\newcounter{remark}
\def\remark#1{
 \refstepcounter{remark}
 \trivlist\item
 \noindent\textbf{Remark \theremark .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\endremark{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Changed 24.01.2012
\newcounter{definition}
\def\definition#1{
 \refstepcounter{definition}
 \trivlist\item
 \noindent\textbf{Definition \thedefinition .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\enddefinition{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Added 10.01.2012
\newcounter{problem}
\def\problem#1{
 \refstepcounter{problem}
 \trivlist\item
 \noindent\textbf{Problem \theproblem .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\endproblem{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Observation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Added 12.11.2012
\newcounter{observation}
\def\observation#1{
 \refstepcounter{observation}
 \trivlist\item
 \noindent\textbf{Observation \theobservation .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\endobservation{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Assumption %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Environment added 22.10.2014
\newcounter{assumption}
\def\assumption#1{
 \refstepcounter{assumption}
 \trivlist\item
 \noindent\textbf{Assumption \theassumption .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\endassumption{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Property %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Environment added 22.10.2012
\newcounter{property}
\def\property#1{
 \refstepcounter{property}
 \trivlist\item
 \noindent\textbf{Property \theproperty .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (\textit{#1}) \fi
}
\def\endproperty{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%% Statement %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Environment added 29.07.2016
\newcounter{statement}
\def\statement#1{
 \refstepcounter{statement}
 \trivlist\item
 \noindent\textbf{Statement \thestatement .}
 \def\@temp{#1}\relax
 \ifx \@temp\@empty \else (#1) \fi
 \it
}
\def\endstatement{\endtrivlist\unskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%% Biography %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Added 19.03.2009
%% Changed 14.12.2009
%% Changed 19.01.2012
\newdimen\@npoints
\newcount\@nbrak
\newcount\@nlines
\newcount\@tekst
\@nlines=10

\def\biography[#1]#2{
\vskip20pt
\parskip=0pt
\interlinepenalty500
\unitlength 1cm
\footnotesize
\def\@temp{#1}
\ifx\@temp\@empty
 \noindent\textbf{#2}
\else
 \def\@tempbox{\begin{minipage}[b][3.2cm][c]{2.5cm}
  \includegraphics[height=3.2cm,width=2.5cm]{#1}
  \end{minipage}}
 \penalty\interlinepenalty
 \clubpenalty4000
 \@tekst\pagegoal
 \advance\@tekst-\pagetotal
 \ifnum\@tekst<5966975
   \newpage
 \fi
 \hangindent2.6cm
 \hangafter-\@nlines
 \noindent\makebox[0pt][l]{\hspace{-2.6cm}\raisebox{-3cm}[0pt][0pt]{\@tempbox}}
 \noindent\hspace{-1mm}\textbf{#2}	
\fi
}

\def\endbiography{
\relax\par
\@nbrak=\prevgraf
\ifnum \@nbrak < 10
 \advance\@nlines-\@nbrak
 \@npoints=\baselineskip
 \multiply\@npoints by \@nlines
 \vskip\@npoints
\fi
}
%%%%%%%%%%%%%%%%%%%%%% Acknowledgment %%%%%%%%%%%%%%%%%%%%%%%%
%% Added 17.07.2009
\newenvironment{acknowledgment}{
  \section*{\centering Acknowledgment}
}

%%%%%%%%%%%%%%%%%%%%%% Appendix %%%%%%%%%%%%%%%%%%%%%%%%
%% Environment appendix added 17.07.2009
%% Environment appendices added 14.12.2009
%% Modified 19.01.2012 (appendix title as parameter)
%% Environment appendix changed 17.09.2012
%% Environments appendix and appendices changed 22.10.2014
\newcounter{appendix}
\renewcommand{\theappendix}{\Alph{appendix}}
\newenvironment{appendices}[1]{
  \refstepcounter{appendix}
  \def\@temp{#1}\relax
  \def\@znak{}\relax
  \section*{\centering Appendix \theappendix}
  \ifx\@temp\@znak \relax
  \else {\centering \large \textbf{#1}}\vspace{7pt} 
  \fi
  %Added 17.09.2012
  \setcounter{equation}{0}\renewcommand{\theequation}{\theappendix\arabic{equation}}
  \setcounter{figure}{0}\renewcommand{\thefigure}{\theappendix\arabic{figure}}
  \setcounter{table}{0}\renewcommand{\thetable}{\theappendix\arabic{table}}
  \setcounter{theorem}{0}\renewcommand{\thetheorem}{\theappendix\arabic{theorem}}
  \setcounter{lemma}{0}\renewcommand{\thelemma}{\theappendix\arabic{lemma}}
  \setcounter{proposition}{0}\renewcommand{\theproposition}{\theappendix\arabic{proposition}}
  \setcounter{corollary}{0}\renewcommand{\thecorollary}{\theappendix\arabic{corollary}}
  \setcounter{example}{0}\renewcommand{\theexample}{\theappendix\arabic{example}}
  \setcounter{remark}{0}\renewcommand{\theremark}{\theappendix\arabic{remark}}
  \setcounter{definition}{0}\renewcommand{\thedefinition}{\theappendix\arabic{definition}}
  \setcounter{problem}{0}\renewcommand{\theproblem}{\theappendix\arabic{problem}}
  %Added 22.10.2014
  \setcounter{assumption}{0}\renewcommand{\theassumption}{\theappendix\arabic{assumption}}
  \setcounter{property}{0}\renewcommand{\theproperty}{\theappendix\arabic{property}}
  \setcounter{section}{0}\renewcommand{\thesection}{\theappendix\arabic{section}}
 %Added 19.01.2018
\setcounter{algorithm}{0}\renewcommand{\thealgorithm}{\theappendix\arabic{algorithm}}
}

\renewenvironment{appendix}[1]{
  \def\@temp{#1}\relax
  \def\@znak{}\relax
  \section*{\centering Appendix}
  \ifx\@temp\@znak \relax
  \else {\centering \large \textbf{#1}}\vspace{7pt} % Modified 29.07.2016
  \fi
  %Added 17.09.2012
  \setcounter{equation}{0}\renewcommand{\theequation}{A\arabic{equation}}
  \setcounter{figure}{0}\renewcommand{\thefigure}{A\arabic{figure}}
  \setcounter{table}{0}\renewcommand{\thetable}{A\arabic{table}}
  \setcounter{theorem}{0}\renewcommand{\thetheorem}{A\arabic{theorem}}
  \setcounter{lemma}{0}\renewcommand{\thelemma}{A\arabic{lemma}}
  \setcounter{proposition}{0}\renewcommand{\theproposition}{A\arabic{proposition}}
  \setcounter{corollary}{0}\renewcommand{\thecorollary}{A\arabic{corollary}}
  \setcounter{example}{0}\renewcommand{\theexample}{A\arabic{example}}
  \setcounter{remark}{0}\renewcommand{\theremark}{A\arabic{remark}}
  \setcounter{definition}{0}\renewcommand{\thedefinition}{A\arabic{definition}}
  \setcounter{problem}{0}\renewcommand{\theproblem}{A\arabic{problem}}
  %Added 22.10.2014
  \setcounter{assumption}{0}\renewcommand{\theassumption}{A\arabic{assumption}}
  \setcounter{property}{0}\renewcommand{\theproperty}{A\arabic{property}}
  \setcounter{section}{0}\renewcommand{\thesection}{A\arabic{section}}
 %Added 19.01.2018
\setcounter{algorithm}{0}\renewcommand{\thealgorithm}{A\arabic{algorithm}}
}


%%%%%%%%%%%%%%%%%%%% Editioral %%%%%%%%%%%%%%%%%%
%% Added 14.12.2009
\newcommand{\editorial}[1]{
 \section*{\Large \centering \MakeUppercase{#1}}\vskip20pt
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

