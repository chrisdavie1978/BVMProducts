import Image from "next/image";

const Hero = () => {
  return (
    <>
      <div className="text-4xl leading-[64px] tracking-tight max-md:max-w-full">
        Interact with
      </div>
      <div className="mt-2 mx-auto flex flex-wrap items-start md:items-center justify-center gap-5 self-start whitespace-nowrap text-7xl font-bold tracking-tight max-md:flex-wrap max-md:text-4xl">
        <div className="md:mt-2.5">
          <Image
            src={"./logo.svg"}
            width={90}
            height={70}
            alt={"Formica Logo"}
          />
        </div>
        <div className="max-md:text-4xl">AI</div>
        <div className="max-md:text-4xl">Engage</div>
      </div>
      <div className="mt-10 text-xl text-left md:text-center tracking-wide max-md:mt-10 max-md:max-w-full">
        Don’t know where to start? Here are some recommendations...
      </div>
    </>
  );
};

export default Hero;
