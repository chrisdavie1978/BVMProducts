import Card from "./card";
import Hero from "./hero";

const Landing = () => {
  return (
    <div>
      <Hero />
      <div className="m-auto mt-5 flex gap-5 max-md:mt-10 max-md:max-w-full flex-wrap max-md:pr-5 min-w-40 md:justify-center md:max-w-[980px]">
        <Card
          boldContent="Come up with concepts"
          normalContent="for a retro style arcade game"
        />
        <Card
          boldContent="Share Advantages of Artificial Intelligence"
          normalContent=""
        />
        <Card
          boldContent="Explain why popcorn pops"
          normalContent="to a kid who loves to watch in the microwave"
        />
        <Card
          boldContent="Plan a trip"
          normalContent="to see the northern lights in Norway"
        />
        <Card
          boldContent="Give me ideas"
          normalContent="for what to wear for a corporate event"
        />
        <Card
          boldContent="Share Advantages of Artificial Intelligence"
          normalContent=""
        />
      </div>
    </div>
  );
};

export default Landing;
